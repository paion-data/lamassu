import unittest

from mlflow.parser import convert_to_knowledge_graph_spec


class TestParser(unittest.TestCase):

    def test_parser(self):
        model_results: list = [
            [
                [['我', 'ARG0', 0, 1], ['爱', 'PRED', 1, 2], ['中国', 'ARG1', 2, 3]]
            ]
        ]

        expected_nodes = [
            '我',
            '中国'
        ]

        expected_links = ['爱']

        assert [node["fields"]["name"] for node in
                convert_to_knowledge_graph_spec(model_results)["nodes"]] == expected_nodes
        assert [node["fields"]["type"] for node in
                convert_to_knowledge_graph_spec(model_results)["links"]] == expected_links
