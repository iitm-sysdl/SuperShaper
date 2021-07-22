from transformers import AutoConfig, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from custom_layers.custom_bert import BertModel, BertForSequenceClassification


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
# new_config.num_hidden_layers = 6
# new_config.sample_num_hidden_layers = 6
# new_config.sample_hidden_size = 720
# new_config.sample_intermediate_size[2] = 3000
# new_config.sample_num_attention_heads[4] = 6
# print(new_config)

from train import sample_subtransformer

new_config = sample_subtransformer(False, True, 10)

new_config.num_hidden_layers = new_config.sample_num_hidden_layers

model.set_sample_config(new_config)

sub_net = model.get_active_subnet(new_config)


def print_subtransformer_config(config):
    print("===========================================================")
    print("hidden size: ", config.sample_hidden_size)
    print("num attention heads: ", config.sample_num_attention_heads)
    print("intermediate sizes: ", config.sample_intermediate_size)
    print("num hidden layers: ", config.sample_num_hidden_layers)
    print("===========================================================")


print_subtransformer_config(new_config)
print("===================================")


state_dict = sub_net.state_dict()

for key in state_dict:
    print(key, state_dict[key].shape)

# print(sub_net)

# for name, param in sub_net.named_parameters():
#    print(name, param.shape)
#

model(**inputs)
sub_net(**inputs)

# print("running seq classification subtransformer for a sample input")
# model(**inputs)
