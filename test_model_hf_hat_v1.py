from transformers import AutoConfig, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from custom_bert import BertModel, BertForSequenceClassification


def add_sampling_params(config):
    config.sample_hidden_size = config.hidden_size
    config.sample_num_attention_heads = config.num_attention_heads
    config.sample_intermediate_size = config.intermediate_size
    config.sample_num_hidden_layers = config.num_hidden_layers
    return config


# modifying config
bert_config = AutoConfig.from_pretrained("bert-base-uncased")
bert_config = add_sampling_params(bert_config)
# preparing inputs
sentences = ["hello how are you", "i am fine"]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer.encode_plus(sentences, return_tensors="pt")

# prepring model
#model = BertModel(config=bert_config)
##model.set_sample_config(bert_config)
#print("running supertransformer for a sample input")
#model(**inputs)


#bert_config.sample_hidden_size = 756  # 768 -> 756
#bert_config.sample_intermediate_size = 3000  # 3072 -> 756
#bert_config.sample_num_hidden_layers = 6  # 12 -> 6
#bert_config.sample_num_attention_heads = 6  # 12 -> 6
#model.set_sample_config(bert_config)
#
#print("running subtransformer for a sample input")
#model(**inputs)


# prepring model
bert_config = AutoConfig.from_pretrained("bert-base-uncased")
bert_config = add_sampling_params(bert_config)
model = BertForSequenceClassification(config=bert_config)
#model.set_sample_config(bert_config)
#print("running seq classification supertransformer for a sample input")
#model(**inputs)

super_bert_config = [] 
for i in range(bert_config.sample_num_hidden_layers+4): # For global config, BERT Embeddings, BERTEncoder layers, BERT Pooler, Classifier
    new_config = AutoConfig.from_pretrained("bert-base-uncased")
    new_config.sample_hidden_size = 756 
    new_config.sample_intermediate_size = 3000 
    new_config.sample_num_hidden_layers = 6
    new_config.sample_num_attention_heads = 6
    super_bert_config.append(new_config)


#bert_config.sample_hidden_size = 756
#bert_config.sample_intermediate_size = 3000
#bert_config.sample_num_hidden_layers = 6
#bert_config.sample_num_attention_heads = 6
model.set_sample_config(super_bert_config)

print("running seq classification subtransformer for a sample input")
model(**inputs)
