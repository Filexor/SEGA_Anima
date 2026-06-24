from comfy_api.latest import ComfyExtension

from typing_extensions import override

from .SEGA_Anima import File_x_SEGA_Anima_

class File_x_SEGA_Anima(ComfyExtension):
    @override
    async def get_node_list(self):
        return [File_x_SEGA_Anima_]
    
async def comfy_entrypoint():
    return File_x_SEGA_Anima()