from casymda.blocks.block_components.block import Block


def block_with_name(blocks: list[Block], name: str) -> Block:
    for block in blocks:
        if block.name == name:
            return block
    raise ValueError(f"no block with name {name} in list {blocks}")
