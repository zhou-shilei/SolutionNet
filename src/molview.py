from __future__ import annotations

"""
molview.py
- Render a static 3D ball-and-stick image in Gradio without Jupyter extensions.

Why not py3Dmol.show()?
- In JupyterLab, py3Dmol may require the jupyterlab_3dmol extension and otherwise renders a warning panel.
- For a stable public web demo, we generate an <iframe srcdoc="..."> that loads 3Dmol.js directly.

This file only depends on RDKit (to generate a 3D MolBlock) and standard Python libs.
"""

from functools import lru_cache
from html import escape as html_escape
import json

from rdkit import Chem
from rdkit.Chem import AllChem


# Multiple CDN fallbacks (some networks block certain CDNs)
THREEDMOL_CDN_URLS = [
    # cdnjs (version pinned)
    "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.5.3/3Dmol-min.js",
    # jsDelivr (npm)
    "https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js",
    # unpkg (npm)
    "https://unpkg.com/3dmol@2.5.4/build/3Dmol-min.js",
    # official (may change / be blocked in some networks)
    "https://3Dmol.org/build/3Dmol-min.js",
    "https://3dmol.csb.pitt.edu/build/3Dmol-min.js",
]


def _is_water(smiles: str) -> bool:
    s = (smiles or "").strip().replace(" ", "")
    return s in {"O", "[OH2]", "[H]O[H]", "o"}


@lru_cache(maxsize=512)
def smiles_to_3d_block(smiles: str, keep_hs: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    ok = AllChem.EmbedMolecule(mol, params)
    if ok != 0:
        ok = AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
        if ok != 0:
            raise ValueError("3D embedding failed.")
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)

    if not keep_hs:
        mol = Chem.RemoveHs(mol)

    return Chem.MolToMolBlock(mol)


def render_ball_stick_html(smiles: str, height: int = 300) -> str:
    """
    Return HTML for Gradio display.

    Render once with 3Dmol.js, then replace the interactive canvas with a static PNG.
    """
    smiles = (smiles or "").strip()
    if not smiles:
        return "<div style='color:#64748b'>Empty</div>"

    keep_hs = _is_water(smiles)
    block = smiles_to_3d_block(smiles, keep_hs=keep_hs)

    # Escape MolBlock for JS string
    molblock_js = json.dumps(block)

    urls_js = json.dumps(THREEDMOL_CDN_URLS)

    srcdoc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  html, body {{ margin:0; padding:0; width:100%; height:100%; background:#ffffff; }}
  #wrap {{ width:100%; height:100%; position:relative; }}
  #viewer {{ width:100%; height:100%; }}
  #msg {{
    position:absolute; left:0; top:0; right:0; bottom:0;
    display:flex; align-items:center; justify-content:center;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
    font-size: 13px;
    color:#475569;
    padding: 12px;
    text-align:center;
  }}
</style>
</head>
<body>
<div id="wrap">
  <div id="viewer"></div>
  <div id="msg">Loading 3D viewer…</div>
</div>

<script>
(function() {{
  const urls = {urls_js};
  const molblock = {molblock_js};

  function init() {{
    try {{
      const msg = document.getElementById("msg");
      msg.style.display = "none";

      const element = document.getElementById("viewer");
      const viewer = $3Dmol.createViewer(element, {{ backgroundColor: "white" }});
      viewer.addModel(molblock, "sdf");
      viewer.setStyle({{}}, {{ stick: {{ radius: 0.18 }}, sphere: {{ scale: 0.28 }} }});
      viewer.zoomTo();
      viewer.render();
      const snapshot = document.createElement("img");
      snapshot.src = viewer.pngURI();
      snapshot.alt = "3D molecular structure";
      snapshot.draggable = false;
      snapshot.style.cssText = "width:100%;height:100%;object-fit:contain;display:block;";
      document.getElementById("wrap").replaceChildren(snapshot);
    }} catch (e) {{
      const msg = document.getElementById("msg");
      msg.style.display = "flex";
      msg.textContent = "3D render error: " + e;
    }}
  }}

  function loadScript(i) {{
    if (window.$3Dmol) {{
      init();
      return;
    }}
    if (i >= urls.length) {{
      const msg = document.getElementById("msg");
      msg.style.display = "flex";
      msg.textContent = "Failed to load 3Dmol.js. Your network may block the CDN.";
      return;
    }}
    const s = document.createElement("script");
    s.src = urls[i];
    s.async = true;
    s.onload = init;
    s.onerror = function() {{ loadScript(i+1); }};
    document.head.appendChild(s);
  }}

  loadScript(0);
}})();
</script>
</body>
</html>"""

    # Use iframe srcdoc to ensure scripts execute regardless of outer-page sanitization
    iframe = (
        f"<iframe title='3D molecular structure: {html_escape(smiles, quote=True)}' "
        f"style='width:100%; height:{int(height)}px; border:1px solid #e5e7eb; border-radius:14px; background:#fff;' "
        f"srcdoc=\"{html_escape(srcdoc, quote=True)}\"></iframe>"
    )
    return iframe
