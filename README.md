# Open Generative Fill

A repository for hacking Generative Fill with Open Source Tools

[Try out the HF Sapace](https://huggingface.co/spaces/open-gen-fill/open-gen-fill-v1)

## Text Based Image Editing

Creating an Open Source alternative for GenerativeFill and other editing tools.

### Through this notebook we aim to: 

- Provide an Edit Prompt
- Provide an Image
- Edit the Image based on the initial Prompt

### How do we propose to do this?

- Accept an `edit_prompt` and an `image` as inputs
- Pass the `edit_prompt` through a language model to extract source and target entities
- Use a Vision Model to `caption` the image
- Create a `replacement_caption` where the source entity of the original image is swapped with target entity in the `edit_prompt`
- Use the source entity to create a segmentation mask using Grounding Dino and SAM
- Use the mask and the `replacement_caption` for image inpainting

The pipeline is shown in the figure below:

![Image](https://github.com/PyImageSearch/blog_posts/assets/44690292/fe2eb574-b8bb-4954-b918-c8bf93974b1b)

## TO-DOs

This repository is still in its early stage and will require some additional work.

- Better Captioning from Vision Model
- Prompt upsampling using Language Model
- More complex editing tasks than replacement
- Optimization


## Citing Generative Fill

If Generative Fill helps your research, we appreciate your citations. Here is the BibTeX entry:

```
@misc{raha2024generativefill,
  title={GenerativeFill with Keras and Diffusers},
  author={Raha, Ritwik and Roy Gosthipaty, Aritra},
  year={2024},
  howpublished={\url{https://github.com/ritwikraha/GenerativeFill-with-Keras-and-Diffusers}},
}
```



