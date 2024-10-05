import hanlp
import mlflow.pyfunc
import pandas
from parser import convert_to_knowledge_graph_spec


class HanLPner(mlflow.pyfunc.PythonModel):

    def __init__(self):
        self.HanLP = None

    def load_context(self, context):
        HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
        self.HanLP = HanLP

    def predict(self, context, model_input):
        texts = []
        for _, row in model_input.iterrows():
            texts.append(row["text"])

        return pandas.Series(convert_to_knowledge_graph_spec(self.HanLP(texts)["srl"]))

if __name__ == '__main__':
    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
            'python=3.10.7',
            'pip',
            {
                'pip': [
                    'mlflow',
                    'mlflow-skinny',
                    'mlflow[extras]',
                    'pandas=={}'.format(pandas.__version__),
                    'hanlp[amr, fasttext, full, tf]'
                ],
            },
        ],
        'name': 'HanLPner'
    }

    # Save the MLflow Model
    mlflow_pyfunc_model_path = "models/HanLPner"
    mlflow.pyfunc.save_model(path=mlflow_pyfunc_model_path, python_model=HanLPner(), conda_env=conda_env)

    loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

    test_data = pandas.DataFrame(
        {
            "text": [
                "我爱中国"
            ]
        }
    )

    test_predictions = loaded_model.predict(test_data)
    print(test_predictions.to_markdown())