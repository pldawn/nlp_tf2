from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ModelLoader:
    def __init__(self, **kwargs):
        self.tokenizer = None
        self.model = None
        self.is_load = False
        self.kwargs = kwargs

    def load(self, model_tag):
        self.tokenizer = AutoTokenizer.from_pretrained(model_tag)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_tag)
        self.is_load = True

    def encode(self, sentences):
        if not self.is_load:
            raise AttributeError("model hasn't be loaded.")

        result = self.tokenizer.encode(sentences)

        return result

    def predict(self, inputs):
        if not self.is_load:
            raise AttributeError("model hasn't be loaded.")

        result = self.model.predict(inputs)

        return result


def main():
    model_tag = "google/pegasus-newsroom"
    agent = ModelLoader()
    agent.load(model_tag)


if __name__ == "__main__":
    main()

