from transformers import AutoConfig, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from custom_bert import BertModel, BertForSequenceClassification


def add_sampling_params(config):
    config.sample_num_hidden_layers = config.num_hidden_layers
    config.sample_hidden_size = config.hidden_size
    config.sample_num_attention_heads = [
        config.num_attention_heads
    ] * config.sample_num_hidden_layers
    config.sample_intermediate_size = [
        config.intermediate_size
    ] * config.sample_num_hidden_layers

    return config


# preparing inputs
sentences = ["hello how are you", "i am fine"]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer.encode_plus(sentences, return_tensors="pt")


# prepring model
bert_config = AutoConfig.from_pretrained("bert-base-uncased")
bert_config = add_sampling_params(bert_config)
print(bert_config)
model = BertForSequenceClassification(config=bert_config)
model.set_sample_config(bert_config)
print("running seq classification supertransformer for a sample input")
model(**inputs)


new_config = AutoConfig.from_pretrained("bert-base-uncased")
new_config = add_sampling_params(new_config)
new_config.sample_num_hidden_layers = 6
new_config.sample_hidden_size = 720
new_config.sample_intermediate_size[2] = 3000
new_config.sample_num_attention_heads[4] = 6
print(new_config)
model.set_sample_config(new_config)

print("running seq classification subtransformer for a sample input")
model(**inputs)
