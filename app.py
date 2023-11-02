import json
import numpy as np
import torch
from transformers import pipeline
import base64
from io import BytesIO


class InferlessPythonModel:
  def initialize(self):
      self.generator = pipeline("text-generation", model="codellama/CodeLlama-34b-Python-hf", device_map="auto")

  def infer(self, inputs):
    prompt = inputs["prompt"]
    pipeline_output = self.generator(prompt)
    return {"generated_result": pipeline_output}

  def finalize(self,args):
    self.generator = None
