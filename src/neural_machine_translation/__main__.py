from neural_machine_translation.preprocess.dataloader import master_loader


def get_texts(file_path: str, max_percent: int = 10) -> str:
    with open(file_path) as f:
        text = f.read()

    size = len(text)
    return text[: int(size * max_percent // 100)]


def create_vocabulary(text_content: str) -> None:
    train_iterator, val_iterator, src_len, trg_len, trg_vocab = master_loader(
        train, val, 2, src_lang, trg_lang, batch_size
    )

    return None


if __name__ == "__main__":
    FILE_PATH = "/Users/saranyapal/Developer/Neural-Machine-Translation/data/wat-2020/en-bn/dev.en"

    content = get_texts(FILE_PATH, max_percent=10)
