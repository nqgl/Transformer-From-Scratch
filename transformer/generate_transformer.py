import transformer
from ..traindata import tokenizer

def main():
    import transformer
    model1600 = transformer.most_recent_model("/home/g/learn/dl/torch/models/sspeare_ztransformer_1784.128-4_512-0.1-6_epoch4.pt")
    model800 = transformer.most_recent_model("/home/g/learn/dl/torch/models/sspeare_ztransformer_1779.128-4_512-0.1-6_epoch49.pt")
    model_before = transformer.most_recent_model("/home/g/learn/dl/torch/models/sspeare_ztransformer_2121*")
    model_moby = transformer.most_recent_model("/home/g/learn/dl/torch/models/sspeare_ztransformer_2191*.pt")
    model_hhgttg = transformer.most_recent_model("/home/g/learn/dl/torch/models/sspeare_ztransformer_2329*.pt")
    import os
    model = model_hhgttg
    test_str = "The Heart of Gold's"
    tokenizer.sample_transformer_generate_mode(model, test_str, length=3100, temperature=0.8, topk=10)
    tokenizer.sample_transformer_generate_mode(model, input("prompt?:\n"), length=800, temperature=0, topk=10)

if __name__ == "__main__":
    main()

