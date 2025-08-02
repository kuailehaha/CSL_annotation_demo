"""
Gradio web app for sign-language action lookup
---------------------------------------------
* 输入：一个中文词语（如“解雇”）
* 输出：表格列
    1) 词语
    2) 动作 ID
    3) 视频预览
    4) 复制按钮
"""

import os, time, json, pickle
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import gradio as gr

# ----------------------------------------------------------------------
# ★★★ 1.   向量检索相关  ★★★
# ----------------------------------------------------------------------
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Qwen Embedding 的末 token pool"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(attention_mask.shape[0], device=attention_mask.device), lengths]

tokenizer = AutoTokenizer.from_pretrained("/2023234343/Qwen3-Embedding-8B", padding_side="left")
model     = AutoModel.from_pretrained("/2023234343/Qwen3-Embedding-8B", torch_dtype=torch.float16).cuda().eval()

def embed_texts(texts, max_length=512):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        emb = last_token_pool(model(**batch).last_hidden_state, batch["attention_mask"])
    return F.normalize(emb, p=2, dim=1)

def get_vocab() -> List[Dict]:
    with open("20250714v2_preview_modified.json", encoding="utf8") as f:
        return json.load(f)

CACHE_PATH   = "doc_embeds.pkl"
COUNTER_PATH = "copy_counter.pkl"

def build_or_load_index(documents):
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        print(f"[INFO] 载入缓存：{CACHE_PATH} (docs={len(data['docs'])})")
        return data["embeds"], data["docs"]

    docs_str  = [d["含义名称"] for d in documents]
    all_embeds = torch.cat([embed_texts(docs_str[i:i+1024]) for i in range(0, len(docs_str), 1024)]).cpu()
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"embeds": all_embeds, "docs": documents}, f)
    print(f"[INFO] 向量缓存已保存：{CACHE_PATH}")
    return all_embeds, documents

def search_by_embedding(keyword: str, doc_embeds: torch.Tensor, documents: List[dict], topk=80):
    sims = torch.matmul(embed_texts([keyword]), doc_embeds.T).squeeze(0)
    k    = min(topk, len(documents))
    topk_scores, idx = torch.topk(sims, k)
    return [documents[i] | {"score": float(topk_scores[j])} for j, i in enumerate(idx)]

doc_embeds, documents = build_or_load_index(get_vocab())
doc_embeds = doc_embeds.to(model.device)

def call_model(query: str):
    docs = search_by_embedding(query, doc_embeds, documents)
    with open("20250714v2_merged_with_keys.json", encoding="utf8") as f:
        meta = json.load(f)
    res = []
    for d in docs:
        item = meta[d["动作id"]]
        if item not in res:
            res.append(item)
    return res

# ----------------------------------------------------------------------
# ★★★ 2.   复制计数持久化  ★★★
# ----------------------------------------------------------------------
def load_copy_counter() -> Dict[str, int]:
    if os.path.exists(COUNTER_PATH):
        with open(COUNTER_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_copy_counter(counter: Dict[str, int]):
    with open(COUNTER_PATH, "wb") as f:
        pickle.dump(counter, f)

def update_copy_count(action_id: str):
    counter = load_copy_counter()
    counter[action_id] = counter.get(action_id, 0) + 1
    save_copy_counter(counter)
    print(f"[实时更新] 动作ID {action_id} 复制次数: {counter[action_id]}")
    return counter[action_id]

# ----------------------------------------------------------------------
# ★★★ 3.   Gradio UI  ★★★
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 2)   前端 JS  (无计数)
# ----------------------------------------------------------------------
COPY_JS = """
<script>
function copyToClipboard(actionId){
  const text='('+actionId+')';
  navigator.clipboard?.writeText(text).then(showToast)
    .catch(()=>{       // 兼容旧浏览器
      const t=document.createElement('textarea');
      t.value=text; t.style.position='fixed'; t.style.opacity=0;
      document.body.appendChild(t); t.select();
      if(document.execCommand('copy')) showToast();
      document.body.removeChild(t);
    });
  function showToast(){
    const toast=document.createElement('div');
    toast.textContent='已复制 ('+actionId+')';
    toast.style='position:fixed;top:20px;right:20px;background:#4CAF50;'
               +'color:#fff;padding:8px 12px;border-radius:4px;z-index:9999;';
    document.body.appendChild(toast);
    setTimeout(()=>toast.remove(),1500);
  }
}
</script>
"""

# ----------------------------------------------------------------------
# 3)   Gradio Blocks
# ----------------------------------------------------------------------
def build_demo():
    style_td = "padding:8px 12px;border:1px solid #ddd;text-align:center;"
    def render_table(data):
        rows = []
        for item in data:
            w   = item.get("词语") or item.get("word") or item.get("含义名称") or ""
            aid = str(item.get("动作id") or item.get("id") or "")
            url = item.get("预览视频地址") or item.get("url") or ""
            rows.append((w, aid, url))

        body = "".join(
            f"<tr style='border-bottom:1px solid #eee;'>"
            f"<td style='{style_td}'>{w}</td>"
            f"<td style='{style_td}'>{aid}</td>"
            f"<td style='{style_td}'>"
            f"<video src='{u}' width='500' height='400' controls "
            f"preload='auto' autoplay loop muted></video></td>"
            f"<td style='{style_td}'>"
            f"<button onclick=\"copyToClipboard('{aid}')\" "
            f"style='padding:6px 12px;background:#4CAF50;color:#fff;"
            f"border:none;border-radius:4px;cursor:pointer;font-size:14px;' "
            f"onmouseover=\"this.style.background='#45a049'\" "
            f"onmouseout=\"this.style.background='#4CAF50'\">复制</button></td></tr>"
            for w, aid, u in rows
        )
        return (
            "<table style='border-collapse:collapse;margin:auto;border:1px solid #ddd'>"
            "<thead><tr style='background:#f2f2f2'>"
            f"<th style='{style_td}'>词语</th><th style='{style_td}'>动作ID</th>"
            f"<th style='{style_td}'>视频预览</th><th style='{style_td}'>复制</th>"
            "</tr></thead><tbody>" + body + "</tbody></table>"
        )

    def run(q: str, prev_t: float) -> Tuple[str, float]:
        now = time.time()
        if now - prev_t < 0.5:
            return "<p>请稍候，查询过于频繁。</p>", prev_t
        data = call_model(q)
        if not data:
            return f"<p>未找到与 <b>{q}</b> 相关的记录。</p>", now
        return render_table(data), now

    with gr.Blocks(title="手语动作查询 Demo", head=COPY_JS) as demo:
        gr.Markdown("## 手语动作查询 Demo\n输入词语后点击“查询”，即可浏览相关手语动作。")
        query  = gr.Textbox(label="输入词语", placeholder="例如：解雇")
        btn    = gr.Button("查询")
        html   = gr.HTML()
        stamp  = gr.State(0.0)
        btn.click(run, [query, stamp], [html, stamp])
        query.submit(run, [query, stamp], [html, stamp])
    return demo

# ----------------------------------------------------------------------

if __name__ == "__main__":
    build_demo().launch(share=True)
