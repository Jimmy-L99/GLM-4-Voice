import argparse
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
import torch
import uvicorn
import sys

from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from vllm.inputs import TokensPrompt
from asyncio.log import logger
from flow_inference import AudioDecoder
import uuid
import os
import io
import soundfile as sf
import base64
import logging
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO)

class BaseResponse(BaseModel):
    type: str
    content: str
    state: Optional[str] = None

class StreamAudioResponse(BaseResponse):
    state: str = "stream"

class CompleteAudioResponse(BaseResponse):
    state: str = "end"

class CompleteTextResponse(BaseResponse):
    pass


app = FastAPI()

def decode_base64_audio(encoded_audio):
    try:
        audio_data = base64.b64decode(encoded_audio)
        audio_bytes = io.BytesIO(audio_data)       
        return audio_bytes
    except:
        raise ValueError("Invalid base64 audio data")

class ModelWorker:
    def __init__(self, model_path, tokenizer_path, flow_path, dtype="bfloat16", device='cuda'):

        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            device=device,
            tensor_parallel_size=2,  
            dtype=dtype, 
            trust_remote_code=True,
            gpu_memory_utilization=0.20,
            enforce_eager=True,
            worker_use_ray=False,
            disable_log_requests=True,
            max_model_len=2048,
        )
        self.device = device
        # GLM
        self.glm_model = AsyncLLMEngine.from_engine_args(engine_args)
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, device=device)
        
        #  Whisper
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(args.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
        
        # decoder
        flow_config = os.path.join(flow_path, "config.yaml")
        flow_checkpoint = os.path.join(flow_path, 'flow.pt')
        hift_checkpoint = os.path.join(flow_path, 'hift.pt')
        self.audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                                          hift_ckpt_path=hift_checkpoint, device=device)        

    
    # tokenizer
    def message_tokenizer(self, prompt, prompt_type, whisper_model, feature_extractor):
        prompt = prompt
        prompt_type = prompt_type

        if prompt_type == "audio":           
            audio_bytes = decode_base64_audio(prompt)
            # 如果输入为音频，将其转换为音频 tokens
            audio_tokens = extract_speech_token(whisper_model, feature_extractor, [audio_bytes])[0]
            
            if len(audio_tokens) == 0:
                raise ValueError("No audio tokens extracted")
            audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
            user_input = audio_tokens
            # system_prompt
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
        else:
            if prompt_type != "text":
                raise ValueError("Input text cannot be None or empty")
            user_input = prompt
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

        inputs = ""
        if "<|system|>" not in inputs:
            inputs += f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        return inputs
    
    @torch.inference_mode()
    async def generate_stream(self, message_tokenizer_results, temperature, top_p, max_new_tokens):
        tokenizer = self.glm_tokenizer
        model = self.glm_model

        inputs = tokenizer([message_tokenizer_results], return_tensors="pt")
        input_ids = inputs['input_ids'][0].tolist()

        params_dict = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.0,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": -1,
            "ignore_eos": False,
            "max_tokens": max_new_tokens,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }
        sampling_params = SamplingParams(**params_dict)
        first_token_time = None
        start_time = time.time()
        async for output in model.generate(
                TokensPrompt(**{
                    "prompt_token_ids": input_ids,
                }),
            sampling_params=sampling_params,
            request_id=f"{time.time()}"
            ):
            if first_token_time is None:
                first_token_time = time.time()
                first_token_latency = first_token_time - start_time
                logger.info(f"Latency for generating first token: {first_token_latency} seconds")
            
            yield int(output.outputs[0].token_ids[-1])

    @torch.inference_mode()
    async def generate_audio_decode(self, generate_stream_output):
        text_tokens, audio_tokens = [], []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        end_token_id = self.glm_tokenizer.convert_tokens_to_ids('<|user|>')
        complete_tokens = []
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self.device)
        this_uuid = str(uuid.uuid4())
        tts_speechs = []
        tts_mels = []
        prev_mel = None
        is_finalize = False
        block_size = 10

        start_decode_time = time.time()
        token_count = 0

        async for token_id in generate_stream_output:
            if token_id == end_token_id:
                is_finalize = True
            if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                block_size = 20
                tts_token = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)

                if prev_mel is not None:
                    prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                # tokens to audio
                tts_speech, tts_mel = self.audio_decoder.token2wav(tts_token, uuid=this_uuid,
                                                                prompt_token=flow_prompt_speech_token.to(self.device),
                                                                prompt_feat=prompt_speech_feat.to(self.device),
                                                                finalize=is_finalize)
                prev_mel = tts_mel

                # save
                tts_speechs.append(tts_speech.squeeze())
                tts_mels.append(tts_mel)    
                flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                audio_tokens = []

                # stream
                audio_data = tts_speech.squeeze().cpu().numpy()
                audio_io = io.BytesIO()
                sf.write(audio_io, audio_data, 22050, format='wav')
                audio_io.seek(0)
                base64_stream_data = base64.b64encode(audio_io.read()).decode('utf-8')
                stream_response = StreamAudioResponse(type="audio", content=base64_stream_data)
                yield json.dumps(stream_response.model_dump()) + "\n"
                # yield json.dumps({"type": "audio", "content": base64_stream_data, "state": "stream"}) + "\n"
                audio_io.truncate(0)
                audio_io.seek(0)

            if not is_finalize:
                complete_tokens.append(token_id)
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)
            
            token_count += 1
        # text
        complete_text = self.glm_tokenizer.decode(complete_tokens, spaces_between_special_tokens=False)
        text_response = CompleteTextResponse(type="text", content=complete_text)
        # yield json.dumps({"type": "text", "content": complete_text}) + "\n"
        yield json.dumps(text_response.model_dump()) + "\n"

        # chunk audio merge
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu().numpy()
        audio_io = io.BytesIO()
        sf.write(audio_io, tts_speech, 22050, format='wav')
        audio_io.seek(0)
        while True:
            data = audio_io.read()
            base64_data = base64.b64encode(data).decode('utf-8')
            if not data:
                break
            complete_audio_response = CompleteAudioResponse(
                type="audio", 
                content=base64_data, 
            )
            yield json.dumps(complete_audio_response.model_dump()) +"\n"
            # yield json.dumps({"type": "audio", "content": base64_data, "state": "end"}) + "\n"
        audio_decode_latency = time.time() - start_decode_time
        tokens_per_second = token_count / audio_decode_latency if audio_decode_latency > 0 else 0
        logger.info(f"Decode efficiency: {tokens_per_second} tokens/second")   
        audio_io.close()


    async def message_handler(self, params):
        prompt = params["prompt"]
        prompt_type = params["prompt_type"]
        temperature = params["temperature"]
        top_p = params["top_p"]
        max_new_tokens = params["max_new_tokens"]
        
        input_tokens = self.message_tokenizer(prompt, prompt_type, self.whisper_model, self.feature_extractor)
        logger.info(f"input_tokens: {input_tokens}")

        model_output = self.generate_stream(input_tokens, temperature, top_p, max_new_tokens)
        
        decoder_output = self.generate_audio_decode(model_output)
        return decoder_output

@app.post("/audio")
async def chatmessage_handler(request: Request):
    params = await request.json()
    decoder_output = await worker.message_handler(params)
    
    return StreamingResponse(decoder_output, media_type="audio/wav")


if __name__ == "__main__":
  
    sys.path.insert(0, "./cosyvoice")
    sys.path.insert(0, "./third_party/Matcha-TTS")

    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-path", type=str, default="glm-4-voice-decoder")
    parser.add_argument("--tokenizer-path", type= str, default="glm-4-voice-tokenizer")
    parser.add_argument("--model-path", type=str, default="glm-4-voice-9b")
    parser.add_argument("--host", type=str, default="host")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    
    worker = ModelWorker(args.model_path, args.tokenizer_path, args.flow_path, args.dtype, args.device)

    uvicorn.run(app, host=args.host, port=args.port)
