import json
import numpy as np
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import base64
from io import BytesIO
import pkg_resources


class InferlessPythonModel:
  def initialize(self):
      self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/CodeLlama-34B-Python-GPTQ", use_fast=True)
      self.model = AutoGPTQForCausalLM.from_quantized(
        "TheBloke/CodeLlama-34B-Python-GPTQ",
        use_safetensors=True,
        device="cuda:0",
        quantize_config=None,
        inject_fused_attention=False
      )

  def infer(self, inputs):
    prompt = inputs["prompt"]
    input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = self.model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    result = self.tokenizer.decode(output[0])
    package_name = "auto-gptq"
    version = self.get_package_version(package_name)
    return {"generated_result": f'version {version}'}

  def get_package_version(self,package_name):
    try:
        # Use pkg_resources to get the distribution
        distribution = pkg_resources.get_distribution(package_name)
        
        # Return the version of the package
        return distribution.version
    except pkg_resources.DistributionNotFound:
        # Handle the case where the package is not installed
        return None

# Example usage for "auto-gptq"

  def finalize(self,args):
    self.tokenizer = None
    self.model = None
