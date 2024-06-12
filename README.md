# Quantized SSMs

This repository contains the modified code and scripts necessary to experiment with quantized State Space Models (SSMs).

We utilize the AQT framework to modify the S5 model architecture to explore the impacts of quantization and mixed precision training on SSMs which is an underexplored question.

Since SSMs are reliant on recurrent dynamics to process information, certain parts of the architecture are less robust to quantization than other parts; specifically, the A matrix which linearly maps the recurrent state of the model over time is sensitive to quantization below 8 bits for tasks in the Long Range Arena, while other parameters in the backbone can survive quantization down to 4 bits or below.

The hope is that the insights from this work will inform future research on quantized SSMs and mixed-precision training and inference for selective SSMs such as Mamba and beyond.
Given that SSMs have linear time and constant spatial complexity with respect to sequence length, quantized SSMs offer promise for future local AI inference with substantially smaller resource demands compared to modern transformer architectures.

# Reproducing Experiments

To reproduce experiments from the paper, use the bash scripts in this repo, which will instantiate and execute a quantized SSM model.