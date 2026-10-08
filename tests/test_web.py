"""Check the Gradio schema and callbacks used by the deployed page."""
import unittest

import app


class WebSmokeTests(unittest.TestCase):
    def test_page_and_api_schema(self):
        demo = app.build_demo()
        self.assertGreater(len(demo.blocks), 0)
        self.assertGreater(len(demo.get_api_info()["named_endpoints"]), 0)
        selectors = [
            component["props"] for component in demo.get_config_file()["components"]
            if component["type"] == "radio"
        ]
        self.assertTrue(any(props.get("value") == "SolutionNet" for props in selectors))

    def test_named_inputs_predict_with_both_models(self):
        for model in ["RandomForest", "SolutionNet"]:
            with self.subTest(model=model):
                _, result = app.ui_predict(
                    model, "乙醇", "水", "THF", app.make_interaction_template(), "", 0.5, True
                )
                self.assertNotIn("error", result)
                self.assertEqual(result["model"], "solutionnet" if model == "SolutionNet" else "rf")
                self.assertGreaterEqual(result["score"], 0)
                self.assertLessEqual(result["score"], 1)
