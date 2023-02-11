import time

from rich import print
from transformers import (PegasusForConditionalGeneration, PegasusTokenizer,
                          T5ForConditionalGeneration, T5TokenizerFast)


class Qgen():
  @staticmethod
  def generate(text: str, num_return_sequences: int = 10, num_beams: int = 10):
    device = 'gpu'
    model_name = 'tuner007/pegasus_paraphrase'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    batch = tokenizer(
      [text],
      truncation=True,
      padding='longest',
      max_length=60, 
      return_tensors="pt"
    ).to(device)

    translated = model.generate(
      **batch,
      max_length=60,
      num_beams=num_beams, 
      num_return_sequences=num_return_sequences, 
      temperature=1.5
    ).to(device)

    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return list(set(tgt_text))

  @staticmethod
  def paraphrase(text: str):
    model_name = 'mrm8488/t5-base-e2e-question-generation'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5TokenizerFast.from_pretrained(model_name)

    try:
      if len(text) < 50:
        raise Exception('input too small')
      generator_args = {'temperature': 100, 'max_length': 100}
      text = 'generate questions: ' + text + ' </s>' # </s> is the EOS token
      input_ids = tokenizer.encode(text, return_tensors='pt')
      res = model.generate(input_ids, **generator_args)
      output = tokenizer.batch_decode(res, skip_special_tokens=True)
      return output
        
    except Exception as e:
      raise e

            
if __name__ == '__main__':
    text = " Oh i use ripgrep daily, its very necessary to check logs"
    print(Qgen.generate(text))