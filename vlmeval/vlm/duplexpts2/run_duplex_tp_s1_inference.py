# %%
import torch
import argparse
from modeling_duplexptS1 import DuplexPTS1Config, DuplexPTS1Model


def get_thinker_embedding(model: DuplexPTS1Model, tok_id):
    emb = model.thinker.model.embed_tokens(torch.tensor(tok_id))
    return emb

def get_perceiver_embedding(model: DuplexPTS1Model, tok_id):
    emb = model.perceiver.model.language_model.embed_tokens(torch.tensor(tok_id))
    return emb


def cal_distance(model: DuplexPTS1Model, tok_id_a, tok_id_b, mode='thinker'):
    if mode == 'thinker':
        embedder = get_thinker_embedding
    elif mode == 'perceiver':
        embedder = get_perceiver_embedding
    else:
        raise NotImplementedError

    emb_a = embedder(model, tok_id_a)
    emb_b = embedder(model, tok_id_b)
    sim = torch.cosine_similarity(emb_a, emb_b, dim=0)
    return sim



# %%

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    PERCEIVER_PATH = "/path/to/home/models/Qwen2.5-VL-3B-Instruct"
    THINKER_PATH = "/path/to/home/models/Qwen3-4B"

    args.add_argument("--perceiver_model", type=str,
                      default="/path/to/home/models/Qwen2.5-VL-3B-Instruct")
    args.add_argument("--thinker_model", type=str,
                      default="/path/to/home/models/Qwen3-4B")
    args.add_argument("--image_path", type=str,
                      default="dataset/flickr30k-images/3451360781.jpg")
    args.add_argument("--question", type=str,
                      default="What color clothes are the left man wearing?")
    args = args.parse_args()

    print(f'init with p={args.perceiver_model} t={args.thinker_model}')

    config = DuplexPTS1Config()
    print(config)

    model = DuplexPTS1Model(config)
    model = model.eval().cuda()

    image = args.image_path
    msgs = [{"role": "user", "content": args.question}]

    image1 = args.image_path
    image2 = args.image_path
    msgs1 = [{"role": "user", "content": 'Is this image bright?'}]
    msgs2 = [{"role": "user", "content": 'Is this image be cropped?'}]

    p_out_ids, t_out_ids, response = model.chat(image,
                                                msgs,
                                                perceiver_generation_params=dict(
                                                    do_sample=True),
                                                thinker_generation_params=dict(
                                                    do_sample=True,
                                                    temperature=0.6,
                                                    top_p=0.95,
                                                    top_k=20,
                                                    min_p=0,
                                                    max_new_tokens=1500
                                                )
                                                )

    # response = model.generate(
    #     [image1, image2],
    #     [msgs1, msgs2],
    #     perceiver_generation_params=dict(
    #         do_sample=True,
    #         max_new_tokens=128),
    #     thinker_generation_params=dict(
    #         do_sample=True,
    #         temperature=0.6,
    #         top_p=0.95,
    #         top_k=20,
    #         min_p=0,
    #         max_new_tokens=15000
    #     )
    # )

    print(f'\n\n## Outside')
    print(f'\n\n## p output:\n{model.t_tokenizer.decode(p_out_ids)}')
    print(f'\n\n## t output:\n{model.t_tokenizer.decode(t_out_ids)}')
    print(f'\n\n## response:\n{response}')
