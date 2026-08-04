# Third-party components

GenET's own source files are Apache-2.0 licensed. No upstream source or model
weights are vendored in this repository.

The production backend is pinned to NVIDIA Cosmos Framework commit
`a904d2d36b774a51dd06ff9ff906816b1a04f579`. Cosmos Framework and Cosmos3
weights are governed by OpenMDW-1.1. Review and retain NVIDIA's license and
notices when redistributing those materials:

- https://github.com/NVIDIA/cosmos-framework
- https://huggingface.co/nvidia/Cosmos3-Edge

The video tokenizer is the Wan2.2 VAE distributed from
`Wan-AI/Wan2.2-TI2V-5B`; review its model-card terms before use.

Architecture references used for the control branch include VACE
(Apache-2.0) and DiffSynth-Studio (Apache-2.0). Their code is not copied here.

