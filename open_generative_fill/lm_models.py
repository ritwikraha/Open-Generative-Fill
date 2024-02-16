import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import delete_model


def run_lm_model(model_id: str, caption: str, edit_prompt: str, device: str = "cuda"):
    language_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        {
            "role": "system",
            "content": "Follow the examples and return the expected output",
        },
        {
            "role": "user",
            "content": "Caption: a blue sky with fluffy clouds\nQuery: Make the sky stormy",
        },
        {
            "role": "assistant",
            "content": "A: sky\nB: a stormy sky with heavy gray clouds, torrential rain, gloomy, overcast",
        },
        {
            "role": "user",
            "content": "Caption: a cat sleeping on a sofa\nQuery: Change the cat to a dog",
        },
        {
            "role": "assistant",
            "content": "A: cat\nB: a dog sleeping on a sofa, cozy and comfortable, snuggled up in a warm blanket, peaceful",
        },
        {
            "role": "user",
            "content": "Caption: a snowy mountain peak\nQuery: Replace the snow with greenery",
        },
        {
            "role": "assistant",
            "content": "A: snow\nB: a lush green mountain peak in summer, clear blue skies, birds flying overhead, serene and majestic",
        },
        {
            "role": "user",
            "content": "Caption: a vintage car parked by the roadside\nQuery: Change the car to a modern electric vehicle",
        },
        {
            "role": "assistant",
            "content": "A: car\nB: a sleek modern electric vehicle parked by the roadside, cutting-edge design, environmentally friendly, silent and powerful",
        },
        {
            "role": "user",
            "content": "Caption: a wooden bridge over a river\nQuery: Make the bridge stone",
        },
        {
            "role": "assistant",
            "content": "A: bridge\nB: an ancient stone bridge over a river, moss-covered, sturdy and timeless, with clear waters flowing beneath",
        },
        {
            "role": "user",
            "content": "Caption: a bowl of salad on the table\nQuery: Replace salad with soup",
        },
        {
            "role": "assistant",
            "content": "A: bowl\nB: a bowl of steaming hot soup on the table, scrumptious, with garnishing",
        },
        {
            "role": "user",
            "content": "Caption: a book on a desk surrounded by stationery\nQuery: Remove all stationery, add a laptop",
        },
        {
            "role": "assistant",
            "content": "A: stationery\nB: a book on a desk with a laptop next to it, modern study setup, focused and productive, technology and education combined",
        },
        {
            "role": "user",
            "content": "Caption: a cup of coffee on a wooden table\nQuery: Change coffee to tea",
        },
        {
            "role": "assistant",
            "content": "A: cup\nB: a steaming cup of tea on a wooden table, calming and aromatic, with a slice of lemon on the side, inviting",
        },
        {
            "role": "user",
            "content": "Caption: a small pen on a white table\nQuery: Change the pen to an elaborate fountain pen",
        },
        {
            "role": "assistant",
            "content": "A: pen\nB: an elaborate fountain pen on a white table, sleek and elegant, with intricate designs, ready for writing",
        },
        {
            "role": "user",
            "content": "Caption: a plain notebook on a desk\nQuery: Replace the notebook with a journal",
        },
        {
            "role": "assistant",
            "content": "A: notebook\nB: an artistically decorated journal on a desk, vibrant cover, filled with creativity, inspiring and personalized",
        },
        {"role": "user", "content": f"Caption: {caption}\nQuery: {edit_prompt}"},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = language_model.generate(
            model_inputs.input_ids, max_new_tokens=512, temperature=0.0, do_sample=False
        )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ]

    output_generation_a, output_generation_b = output_generation.split("\n")
    to_replace = output_generation_a[2:].strip()
    replaced_caption = output_generation_b[2:].strip()

    delete_model(language_model)
    return to_replace, replaced_caption
