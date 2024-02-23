# Open Generative Fill

![YouTube Live Stream](https://www.youtube.com/watch?v=g_4FHoNx0nQ)

A repository for hacking Generative Fill with Open Source Tools

[Try out the HF Sapace](https://huggingface.co/spaces/open-gen-fill/open-gen-fill-v1)

[Try out the scripts](https://colab.research.google.com/gist/ariG23498/dc3530eca187cebe664e66b0ff7082b7/scratchpad.ipynb)

## Text Based Image Editing

Creating an Open Source alternative for GenerativeFill and other editing tools.

### Through this notebook we aim to: 

- Provide an Edit Prompt
- Provide an Image
- Edit the Image based on the initial Prompt

### How do we propose to do this?

- Accept an `edit_prompt` and an `image` as input
- Use a Vision Model to `caption` the image
- Pass the `edit_prompt` through a language model to extract the source entity
- Create a `replacement_caption` where the source entity of the original image is swapped with the target entity in the `edit_prompt`
- Use the source entity to create a segmentation mask using OWL-VIT and SAM
- Use the mask and the `replacement_caption` for image inpainting

The pipeline is shown in the figure below:

![Image](https://i.imgur.com/6xxs6q0.png)

## TO-DOs

This repository is still in its early stages and will require additional work.

- Better Captioning from Vision Model
- Prompt upsampling using the Language Model
- More complex editing tasks than replacement
- Optimization of the models and an end-to-end pipeline

## Credits

- [sayakpaul](https://github.com/sayakpaul) for the amazing advice and ideas
- [pedrogengo](https://github.com/pedrogengo) for the replacement caption idea, as illustrated [here](https://github.com/ritwikraha/Open-Generative-Fill/issues/2#issuecomment-1942670904)
- [rishiraj](https://github.com/rishiraj) for patiently teaching us about Qwen and small LLMs

## Citing Generative Fill

If Generative Fill helps your research, we appreciate your citations. Here is the BibTeX entry:

```
@misc{raha2024opengenerativefill,
  title={Open Source Generative Fill},
  author={Raha, Ritwik and Roy Gosthipaty, Aritra},
  year={2024},
  howpublished={\url{https://github.com/ritwikraha/GenerativeFill-with-Keras-and-Diffusers}},
}
```



