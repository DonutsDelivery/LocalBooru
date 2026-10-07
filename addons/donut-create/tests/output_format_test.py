"""Synthetic queue checks; no ComfyUI instance or user profile is accessed."""

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx


spec = importlib.util.spec_from_file_location(
    "create_output_format_controller", Path(__file__).resolve().parents[1] / "app.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OutputFormatTests(unittest.IsolatedAsyncioTestCase):
    # AC: @donut-create-plugin ac-webp-output
    async def test_queued_outputs_are_lossless_webp_for_both_destinations(self):
        for destination in ("preview", "comfy"):
            with self.subTest(destination=destination):
                controller = module.Controller.__new__(module.Controller)
                session = {"id": "synthetic-studio", "jobs": {}, "backend_url": "http://synthetic.invalid"}
                controller.sessions = {session["id"]: session}
                controller.clock = lambda: 123
                controller.persist = Mock()
                controller.validate_prompt = Mock()
                controller.request_backend = AsyncMock(return_value=httpx.Response(
                    200, json={"prompt_id": "synthetic-job"}))
                workflow = {"nodes": [{"id": 1, "type": "DonutImageSave",
                    "properties": {"dmc_final_output": True}}],
                    "extra": {"dmc_output_destination": destination}}
                original_workflow = copy.deepcopy(workflow)
                body = {"prompt": {"1": {"class_type": "DonutImageSave", "inputs": {
                    "images": ["2", 0], "root": "output", "extension": "png",
                    "lossless_webp": False}}}, "extra_data": {"extra_pnginfo": {"workflow": workflow}}}
                await controller.queue_prompt(session, body)
                sent = controller.request_backend.call_args.kwargs["json"]
                inputs = sent["prompt"]["1"]["inputs"]
                self.assertEqual(inputs["extension"], "webp")
                self.assertTrue(inputs["lossless_webp"])
                self.assertEqual(inputs["images"], ["2", 0])
                self.assertEqual(inputs["root"], "output" if destination == "comfy" else "temp")
                self.assertEqual(sent["extra_data"]["extra_pnginfo"]["workflow"], original_workflow)
                self.assertTrue(session["jobs"]["synthetic-job"]["save_nodes"]["1"])


if __name__ == "__main__":
    unittest.main()
