from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-large-portuguese-cased")
model = AutoModelForMaskedLM.from_pretrained("neuralmind/bert-large-portuguese-cased", device_map="auto")