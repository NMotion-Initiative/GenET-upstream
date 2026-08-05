# Third-party components

GenET's own source files are Apache-2.0 licensed. No upstream source or model
weights are vendored in this repository.

The production backend is pinned to NVIDIA Cosmos Framework commit
`a904d2d36b774a51dd06ff9ff906816b1a04f579`. Cosmos Framework and Cosmos3
weights are governed by OpenMDW-1.1. Review and retain NVIDIA's license and
notices when redistributing those materials:

- https://github.com/NVIDIA/cosmos-framework
- https://huggingface.co/nvidia/Cosmos3-Edge

The reviewed Cosmos3-Edge model snapshot is pinned to Hugging Face revision
`2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2`.

The video tokenizer is the Wan2.2 VAE distributed from
https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B at revision
`921dbaf3f1674a56f47e83fb80a34bac8a8f203e`; review its model-card terms before use.

RoboTwin MDS reading uses MosaicML Streaming `0.13.0`, distributed under
Apache-2.0 from https://github.com/mosaicml/streaming. Its source is not vendored.

Architecture references used for the control branch include VACE
(Apache-2.0) and DiffSynth-Studio (Apache-2.0). Their code is not copied here.
