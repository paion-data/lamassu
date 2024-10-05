import gradio as gr

import hanlp
from mlflow.parser import convert_to_knowledge_graph_spec

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

def inference(input):
    return convert_to_knowledge_graph_spec(HanLP([input])["srl"])

app = gr.Interface(
    fn=inference,
    inputs="text",
    outputs="json",
    title="Named Entity Recognition",
    description=("Turning text corpus into graph representation"),
    examples=[
        ["我爱中国"],
        ["世界会变、科技会变，但「派昂」不会变，它不会向任何人低头，不会向任何困难低头，甚至不会向「时代」低头。「派昂」，永远引领对科技的热爱。只有那些不向梦想道路上的阻挠认输的人，才配得上与我们一起追逐梦想"]
    ],
)
app.launch()
