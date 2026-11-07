"""
Especialização BIM UFPE - Do BIM à Inteligência Artificial: a transformação digital disruptiva na indústria da Construção
Célula BIM UFPE · Professor: Ítalo Guedes

Arquivo autocontido com suporte a carregamento de logo local e modelo IFC de exemplo.

Executar localmente:
    pip install streamlit ifcopenshell pandas numpy plotly matplotlib reportlab pypdf requests
    streamlit run main.py
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import random
import re
import tempfile
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Verificação de dependências (mensagem clara no Streamlit Cloud)
# ---------------------------------------------------------------------------
_REQ = {"plotly": "plotly", "ifcopenshell": "ifcopenshell", "pandas": "pandas", "numpy": "numpy",
        "matplotlib": "matplotlib", "reportlab": "reportlab", "pypdf": "pypdf", "requests": "requests"}
_missing = []
for _mod, _pkg in _REQ.items():
    try:
        __import__(_mod)
    except ModuleNotFoundError:
        _missing.append(_pkg)
if _missing:
    st.set_page_config(page_title="Especialização BIM UFPE", layout="centered")
    st.error("Bibliotecas não instaladas: **" + ", ".join(_missing) + "**")
    st.markdown("""
O servidor não encontrou o arquivo **requirements.txt**. Para corrigir:

1. No repositório do GitHub, crie o arquivo **`requirements.txt`** (nome exato, minúsculas)
   **na mesma pasta deste arquivo .py**, com o conteúdo abaixo.
2. No Streamlit Cloud: *Manage app* → **⋮** → **Reboot app**.
""")
    st.code("streamlit>=1.46\nifcopenshell>=0.8.0\npandas>=2.0\nnumpy\nplotly>=5.20\n"
            "matplotlib>=3.8\nreportlab>=4.0\npypdf>=4.0\nrequests", language="text")
    st.stop()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.material
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.type
import ifcopenshell.api.unit
import ifcopenshell.util.element as ue
import ifcopenshell.util.unit as uu

# Funções auxiliares para carregar a logo (lê de logo_ufpe.png ou input_file_0.png)
def get_logo_bytes() -> bytes | None:
    for filename in ["logo_ufpe.png", "logo celula bim UFPE.png", "input_file_0.png"]:
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    return f.read()
            except Exception:
                pass
    return None

LOGO_BYTES = get_logo_bytes()


# ============================================================================
# EXTRAÇÃO DE DADOS IFC
# ============================================================================

SKIP_CLASSES = (
    "IfcOpeningElement",
    "IfcVirtualElement",
    "IfcFeatureElement",
    "IfcAnnotation",
    "IfcGrid",
)

VOL_KEYS = ["NetVolume", "GrossVolume", "Volume"]
AREA_KEYS = [
    "NetSideArea", "NetArea", "NetFloorArea", "GrossSideArea", "GrossArea",
    "GrossFloorArea", "Area", "NetSurfaceArea", "OuterSurfaceArea", "CrossSectionArea",
]
LEN_KEYS = ["Length", "Span", "Perimeter"]

COLUMNS = [
    "Modelo", "GlobalId", "Classe", "Nome", "Tipo", "Pavimento", "Elevacao",
    "Material", "Externo", "Estrutural", "ResistenciaFogo",
    "Volume_m3", "Area_m2", "Comprimento_m", "Qtd_Psets", "Fonte_Quantidade",
]
CATEGORICAL = ["Modelo", "Classe", "Nome", "Tipo", "Pavimento", "Material",
               "Externo", "Estrutural", "ResistenciaFogo", "Fonte_Quantidade"]
NUMERIC = ["Volume_m3", "Area_m2", "Comprimento_m", "Elevacao", "Qtd_Psets"]


def _find_storey(el, depth: int = 0):
    if depth > 8 or el is None:
        return None
    c = ue.get_container(el)
    if c is None:
        c = ue.get_aggregate(el)
    if c is None:
        return None
    if c.is_a("IfcBuildingStorey"):
        return c
    if c.is_a("IfcSpatialStructureElement") and not c.is_a("IfcSpace"):
        return c
    return _find_storey(c, depth + 1)


def _pick(qtos: dict, keys: list[str]):
    for k in keys:
        for qset in qtos.values():
            v = qset.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    return None


def _yesno(v):
    if v is True:
        return "Sim"
    if v is False:
        return "Não"
    return None


def _common_prop(psets: dict, prop: str):
    for name, vals in psets.items():
        if name.endswith("Common") and prop in vals:
            return vals[prop]
    return None


def _materials(el) -> str | None:
    try:
        mats = ue.get_materials(el)
    except Exception:
        mats = []
    names = sorted({m.Name for m in mats if getattr(m, "Name", None)})
    return " | ".join(names) if names else None


def _geom_quantities(el, settings):
    import ifcopenshell.geom
    import ifcopenshell.util.shape as us
    try:
        shape = ifcopenshell.geom.create_shape(settings, el)
        g = shape.geometry
        vol = us.get_volume(g)
        area = us.get_max_side_area(g)
        return (vol if vol and vol > 0 else None, area if area and area > 0 else None)
    except Exception:
        return (None, None)


def open_ifc_bytes(data: bytes):
    with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return ifcopenshell.open(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def model_info(model) -> dict:
    proj = (model.by_type("IfcProject") or [None])[0]
    app = (model.by_type("IfcApplication") or [None])[0]
    return {
        "Schema": model.schema,
        "Projeto": getattr(proj, "Name", None) or "-",
        "Software": getattr(app, "ApplicationFullName", None) or "-",
        "Pavimentos": len(model.by_type("IfcBuildingStorey")),
    }


def extract(model, model_name: str, geometric_quantities: bool = False,
            geom_limit: int = 3000, progress=None):
    len_scale = uu.calculate_unit_scale(model, "LENGTHUNIT")
    try:
        area_scale = uu.calculate_unit_scale(model, "AREAUNIT")
        vol_scale = uu.calculate_unit_scale(model, "VOLUMEUNIT")
    except Exception:
        area_scale, vol_scale = len_scale ** 2, len_scale ** 3
    elements = [e for e in model.by_type("IfcElement")
                if not any(e.is_a(c) for c in SKIP_CLASSES)]
    elements += list(model.by_type("IfcSpace"))

    settings = None
    if geometric_quantities:
        import ifcopenshell.geom
        settings = ifcopenshell.geom.settings()

    rows, props = [], []
    n = len(elements)
    geom_count = 0
    for i, el in enumerate(elements):
        if progress and (i % 200 == 0 or i == n - 1):
            progress((i + 1) / max(n, 1), f"{model_name}: {i + 1}/{n} elementos")

        try:
            psets = ue.get_psets(el, psets_only=True)
            qtos = ue.get_psets(el, qtos_only=True)
        except Exception:
            psets, qtos = {}, {}

        storey = _find_storey(el)
        elev = None
        if storey is not None and storey.is_a("IfcBuildingStorey") and storey.Elevation is not None:
            elev = round(float(storey.Elevation) * len_scale, 3)
        etype = ue.get_type(el)

        vol = _pick(qtos, VOL_KEYS)
        area = _pick(qtos, AREA_KEYS)
        length = _pick(qtos, LEN_KEYS)
        vol = vol * vol_scale if vol else None
        area = area * area_scale if area else None
        length = length * len_scale if length else None
        source = "IFC (Qto)" if (vol or area or length) else "Sem quantidade"

        if settings is not None and vol is None and geom_count < geom_limit \
                and getattr(el, "Representation", None) is not None:
            gv, ga = _geom_quantities(el, settings)
            geom_count += 1
            if gv or ga:
                vol = vol or gv
                area = area or ga
                source = "Geometria"

        rows.append({
            "Modelo": model_name,
            "GlobalId": el.GlobalId,
            "Classe": el.is_a(),
            "Nome": el.Name or "(sem nome)",
            "Tipo": (etype.Name if etype is not None and etype.Name else None)
                    or getattr(el, "ObjectType", None) or "(sem tipo)",
            "Pavimento": (storey.Name or storey.is_a()) if storey is not None else "(sem pavimento)",
            "Elevacao": elev,
            "Material": _materials(el) or "(sem material)",
            "Externo": _yesno(_common_prop(psets, "IsExternal")) or "(não informado)",
            "Estrutural": _yesno(_common_prop(psets, "LoadBearing")) or "(não informado)",
            "ResistenciaFogo": str(_common_prop(psets, "FireRating") or "(não informado)"),
            "Volume_m3": round(vol, 4) if vol else None,
            "Area_m2": round(area, 4) if area else None,
            "Comprimento_m": round(length, 4) if length else None,
            "Qtd_Psets": len(psets),
            "Fonte_Quantidade": source,
        })

        for set_name, vals in {**psets, **qtos}.items():
            for k, v in vals.items():
                if k == "id" or isinstance(v, (dict, list, tuple)):
                    continue
                props.append((el.GlobalId, set_name, k, v if v is not None else ""))

    df = pd.DataFrame(rows, columns=COLUMNS)
    dprops = pd.DataFrame(props, columns=["GlobalId", "Pset", "Propriedade", "Valor"])
    dprops["Valor"] = dprops["Valor"].astype(str)
    return df, dprops


# ============================================================================
# MODELO IFC DE EXEMPLO
# ============================================================================

def _placement(x, y, z, rot=0.0):
    m = np.eye(4)
    c, s = np.cos(rot), np.sin(rot)
    m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m[:3, 3] = [x, y, z]
    return m


def build_sample_ifc(seed: int = 7) -> bytes:
    rnd = random.Random(seed)
    api = ifcopenshell.api
    f = api.project.create_file(version="IFC4")
    project = api.root.create_entity(f, ifc_class="IfcProject", name="Edifício Didático Célula BIM UFPE")
    api.unit.assign_unit(f, length={"is_metric": True, "raw": "METERS"})
    model_ctx = api.context.add_context(f, context_type="Model")
    body = api.context.add_context(f, context_type="Model", context_identifier="Body",
                                   target_view="MODEL_VIEW", parent=model_ctx)

    site = api.root.create_entity(f, ifc_class="IfcSite", name="Terreno")
    building = api.root.create_entity(f, ifc_class="IfcBuilding", name="Bloco A")
    api.aggregate.assign_object(f, products=[site], relating_object=project)
    api.aggregate.assign_object(f, products=[building], relating_object=site)

    mats = {n: api.material.add_material(f, name=n, category=c) for n, c in [
        ("Concreto C30", "concrete"), ("Alvenaria Bloco Cerâmico", "block"),
        ("Aço CA-50", "steel"), ("Vidro Temperado", "glass"), ("Madeira", "wood"),
        ("Porcelanato", "ceramic")]}

    def mk_type(cls, name):
        return api.root.create_entity(f, ifc_class=cls, name=name)

    types = {
        "ext": mk_type("IfcWallType", "Parede Externa 20cm"),
        "int": mk_type("IfcWallType", "Parede Interna 15cm"),
        "laje": mk_type("IfcSlabType", "Laje Maciça 12cm"),
        "pilar": mk_type("IfcColumnType", "Pilar 30x30"),
        "viga": mk_type("IfcBeamType", "Viga 20x50"),
        "porta": mk_type("IfcDoorType", "Porta Madeira 80x210"),
        "janela": mk_type("IfcWindowType", "Janela Correr 150x120"),
    }

    def element(cls, name, storey, typ, mat, dims, pos, qto_name, qtos, common, rot=0.0):
        el = api.root.create_entity(f, ifc_class=cls, name=name)
        api.spatial.assign_container(f, products=[el], relating_structure=storey)
        api.type.assign_type(f, related_objects=[el], relating_type=typ)
        api.material.assign_material(f, products=[el], type="IfcMaterial", material=mat)
        if dims:
            rep = api.geometry.add_wall_representation(f, context=body, length=dims[0],
                                                       height=dims[1], thickness=dims[2])
            api.geometry.assign_representation(f, product=el, representation=rep)
            api.geometry.edit_object_placement(f, product=el, matrix=_placement(*pos, rot))
        q = api.pset.add_qto(f, product=el, name=qto_name)
        api.pset.edit_qto(f, qto=q, properties=qtos)
        p = api.pset.add_pset(f, product=el, name=f"Pset_{cls[3:]}Common")
        api.pset.edit_pset(f, pset=p, properties=common)
        p2 = api.pset.add_pset(f, product=el, name="UFPE_Planejamento")
        api.pset.edit_pset(f, pset=p2, properties={
            "Etapa": rnd.choice(["Estrutura", "Vedação", "Acabamento"]),
            "Status": rnd.choice(["Executado", "Em execução", "Não iniciado"]),
            "CustoEstimado_R$": round(rnd.uniform(500, 15000), 2),
        })
        return el

    h, W, L = 3.0, 12.0, 20.0
    for level in range(3):
        z = level * h
        st = api.root.create_entity(f, ifc_class="IfcBuildingStorey",
                                    name=["Térreo", "Pavimento 1", "Cobertura"][level])
        st.Elevation = z
        api.aggregate.assign_object(f, products=[st], relating_object=building)

        for i, (ln, x, y, r) in enumerate([(L, 0, 0, 0), (W, L, 0, np.pi / 2),
                                           (L, L, W, np.pi), (W, 0, W, -np.pi / 2)]):
            element("IfcWall", f"PE-{level}{i}", st, types["ext"], mats["Alvenaria Bloco Cerâmico"],
                    (ln, h, 0.2), (x, y, z), "Qto_WallBaseQuantities",
                    {"Length": ln, "Height": h, "Width": 0.2, "NetSideArea": round(ln * h * 0.85, 2),
                     "NetVolume": round(ln * h * 0.2 * 0.85, 3)},
                    {"IsExternal": True, "LoadBearing": False, "FireRating": "EI60"}, rot=r)

        for i in range(rnd.randint(4, 7)):
            ln = round(rnd.uniform(2.5, 6.0), 2)
            element("IfcWall", f"PI-{level}{i}", st, types["int"], mats["Alvenaria Bloco Cerâmico"],
                    (ln, h, 0.15), (rnd.uniform(1, 15), rnd.uniform(1, 10), z), "Qto_WallBaseQuantities",
                    {"Length": ln, "Height": h, "Width": 0.15, "NetSideArea": round(ln * h, 2),
                     "NetVolume": round(ln * h * 0.15, 3)},
                    {"IsExternal": False, "LoadBearing": False, "FireRating": "EI30"})

        element("IfcSlab", f"LJ-{level}", st, types["laje"], mats["Concreto C30"], (L, 0.12, W),
                (0, W, z + h - 0.12), "Qto_SlabBaseQuantities",
                {"Width": 0.12, "NetArea": L * W, "NetVolume": round(L * W * 0.12, 3)},
                {"IsExternal": level == 2, "LoadBearing": True, "FireRating": "REI90"})

        for i, (x, y) in enumerate([(x, y) for x in (0, 10, 20) for y in (0, 6, 12)]):
            element("IfcColumn", f"P{level}{i:02d}", st, types["pilar"], mats["Concreto C30"],
                    (0.3, h, 0.3), (x, y, z), "Qto_ColumnBaseQuantities",
                    {"Length": h, "CrossSectionArea": 0.09, "NetVolume": round(0.09 * h, 3)},
                    {"IsExternal": False, "LoadBearing": True, "FireRating": "R90"})

        for i in range(6):
            ln = 10.0 if i < 4 else 12.0
            element("IfcBeam", f"V{level}{i:02d}", st, types["viga"], mats["Concreto C30"],
                    (ln, 0.5, 0.2), (0, i * 2.4, z + h - 0.5), "Qto_BeamBaseQuantities",
                    {"Length": ln, "CrossSectionArea": 0.1, "NetVolume": round(0.1 * ln, 3)},
                    {"IsExternal": False, "LoadBearing": True, "FireRating": "R90"})

        if level < 2:
            for i in range(rnd.randint(5, 8)):
                element("IfcDoor", f"PT-{level}{i}", st, types["porta"], mats["Madeira"], None, None,
                        "Qto_DoorBaseQuantities", {"Width": 0.8, "Height": 2.1, "Area": 1.68},
                        {"IsExternal": i == 0, "FireRating": "P30" if i == 0 else ""})
            for i in range(rnd.randint(6, 10)):
                element("IfcWindow", f"JN-{level}{i}", st, types["janela"], mats["Vidro Temperado"],
                        None, None, "Qto_WindowBaseQuantities",
                        {"Width": 1.5, "Height": 1.2, "Area": 1.8},
                        {"IsExternal": True})
            for i, (nm, a) in enumerate([("Sala", 38.0), ("Cozinha", 14.5), ("Dormitório 1", 12.0),
                                         ("Dormitório 2", 10.5), ("Banheiro", 4.8)]):
                sp = api.root.create_entity(f, ifc_class="IfcSpace", name=f"{nm} {level}")
                api.aggregate.assign_object(f, products=[sp], relating_object=st)
                q = api.pset.add_qto(f, product=sp, name="Qto_SpaceBaseQuantities")
                api.pset.edit_qto(f, qto=q, properties={"NetFloorArea": a, "Height": 2.8,
                                                        "NetVolume": round(a * 2.8, 2)})
                sp_mat = mats["Porcelanato"]
                api.material.assign_material(f, products=[sp], type="IfcMaterial", material=sp_mat)
    return f.to_string().encode("utf-8")


# ============================================================================
# MOTOR DE GRÁFICOS (specs -> Plotly / Matplotlib)
# ============================================================================

BRAND = {"blue": "#1B60F2", "green": "#42CF96", "navy": "#322F44", "bg": "#F6F8FC"}
PALETTE = ["#1B60F2", "#42CF96", "#322F44", "#F2A541", "#8E6CEF", "#E4572E",
           "#17BEBB", "#7A8CA5", "#B5D33D", "#F06EAA", "#5B8DEF", "#2E9E6E"]

CHART_TYPES = ["bar", "barh", "stacked_bar", "pie", "donut", "treemap",
               "histogram", "scatter", "line", "box", "table", "kpi"]
AGGS = ["count", "sum", "mean", "max", "min", "nunique"]
OPS = ["==", "!=", "in", "not in", ">", ">=", "<", "<=", "contains"]

AGG_PT = {"count": "Quantidade", "sum": "Soma", "mean": "Média", "max": "Máximo",
          "min": "Mínimo", "nunique": "Nº distintos"}
QTY_COLS = ["Volume_m3", "Area_m2", "Comprimento_m"]


def validate_spec(spec: dict, df: pd.DataFrame) -> tuple[dict, list[str]]:
    warn = []
    s = {k: v for k, v in spec.items() if v not in ("", "null")}
    s["type"] = str(s.get("type", "bar")).lower()
    if s["type"] not in CHART_TYPES:
        warn.append(f"Tipo '{s['type']}' inválido; usando 'bar'.")
        s["type"] = "bar"
    for key in ("x", "y", "color"):
        col = s.get(key)
        if col and col not in df.columns:
            warn.append(f"Coluna '{col}' não existe ({key}); ignorada.")
            s[key] = None
    agg = str(s.get("agg") or ("sum" if s.get("y") else "count")).lower()
    if agg not in AGGS:
        agg = "count"
    if s.get("y") and agg not in ("count", "nunique") and not pd.api.types.is_numeric_dtype(df[s["y"]]):
        warn.append(f"'{s['y']}' não é numérica; usando contagem.")
        agg = "count"
    s["agg"] = agg
    if s["type"] not in ("histogram", "kpi", "table") and not s.get("x"):
        s["x"] = "Classe"
    num = lambda c: bool(c) and c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    if s["type"] == "histogram":
        s["x"] = s.get("x") if num(s.get("x")) else (s.get("y") if num(s.get("y")) else "Volume_m3")
    if s["type"] in ("scatter", "box") and not num(s.get("y")):
        s["y"] = "Volume_m3"
    if s["type"] == "scatter" and not num(s.get("x")):
        s["x"] = "Area_m2"
    try:
        s["top_n"] = int(s.get("top_n") or 0)
    except (TypeError, ValueError):
        s["top_n"] = 0
    s["title"] = s.get("title") or _auto_title(s)
    s["filters"] = [f for f in (s.get("filters") or []) if isinstance(f, dict) and f.get("column") in df.columns]
    return s, warn


def _auto_title(s):
    if s["type"] == "kpi":
        return f"{AGG_PT[s['agg']]} de {s.get('y') or 'elementos'}"
    if s["type"] == "histogram":
        return f"Distribuição de {s.get('x')}"
    if s["type"] == "scatter":
        return f"{s.get('y')} × {s.get('x')}"
    if s["type"] == "box":
        return f"Dispersão de {s.get('y')} por {s.get('x')}"
    what = "Elementos" if s["agg"] == "count" else f"{AGG_PT[s['agg']]} de {s.get('y')}"
    by = f" por {s['x']}" if s.get("x") else ""
    return f"{what}{by}"


def apply_filters(df: pd.DataFrame, filters: list[dict] | None) -> pd.DataFrame:
    d = df
    for f in filters or []:
        col, op, val = f.get("column"), f.get("op", "=="), f.get("value")
        if col not in d.columns:
            continue
        s = d[col]
        try:
            if op in ("in", "not in"):
                vals = val if isinstance(val, list) else [val]
                m = s.astype(str).isin([str(v) for v in vals])
                d = d[m] if op == "in" else d[~m]
            elif op == "contains":
                d = d[s.astype(str).str.contains(str(val), case=False, na=False)]
            elif op in ("==", "!="):
                m = s.astype(str) == str(val)
                d = d[m] if op == "==" else d[~m]
            else:
                v = float(val)
                num = pd.to_numeric(s, errors="coerce")
                d = d[{">": num > v, ">=": num >= v, "<": num < v, "<=": num <= v}[op]]
        except Exception:
            continue
    return d


def aggregate(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    d = apply_filters(df, spec.get("filters"))
    x, y, color, agg = spec.get("x"), spec.get("y"), spec.get("color"), spec["agg"]
    if spec["type"] in ("histogram", "scatter", "box"):
        return d
    if spec["type"] == "table":
        keys = [k for k in [x, color] if k]
        if not keys:
            return d
        g = d.groupby(keys, dropna=False)
        out = g.size().rename("Elementos").to_frame()
        for q in QTY_COLS:
            if q in d.columns and d[q].notna().any():
                out[q] = g[q].sum().round(2)
        return out.reset_index().sort_values("Elementos", ascending=False)
    if spec["type"] == "kpi":
        val = len(d) if agg == "count" or not y else getattr(d[y], agg)()
        return pd.DataFrame({"valor": [val]})

    d = d.copy()
    keys = [x] + ([color] if color and color != x else [])
    for k in keys:
        d[k] = d[k].fillna("(sem valor)").astype(str)
    if agg == "count" or not y:
        g = d.groupby(keys).size().reset_index(name="valor")
    else:
        g = d.groupby(keys)[y].agg(agg).reset_index(name="valor")
    g["valor"] = pd.to_numeric(g["valor"], errors="coerce").fillna(0).round(3)

    totals = g.groupby(x)["valor"].sum()
    if spec.get("sort") == "x" or x == "Pavimento":
        if x == "Pavimento" and "Elevacao" in df.columns:
            elev = df.groupby("Pavimento")["Elevacao"].mean()
            order = sorted(totals.index, key=lambda k: (elev.get(k, np.inf) if pd.notna(elev.get(k, np.nan)) else np.inf, k))
        else:
            order = sorted(totals.index)
    else:
        order = list(totals.sort_values(ascending=False).index)
    top = spec.get("top_n") or 0
    if top and len(order) > top:
        keep = set(totals.sort_values(ascending=False).index[:top])
        order = [o for o in order if o in keep]
        g = g[g[x].isin(keep)]
    g[x] = pd.Categorical(g[x], categories=order, ordered=True)
    return g.sort_values(x).reset_index(drop=True)


def value_label(spec):
    if spec["agg"] == "count" or not spec.get("y"):
        return "Elementos"
    return f"{AGG_PT[spec['agg']]} {spec['y']}"


def _layout(fig, spec, height=380):
    fig.update_layout(
        template="plotly_white", height=height, colorway=PALETTE,
        title=dict(text=spec["title"], font=dict(size=15, color=BRAND["navy"]), x=0.01),
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#3c3a52"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


COLOR_MAPS: dict[str, dict] = {}


def set_color_maps(df: pd.DataFrame, cols):
    COLOR_MAPS.clear()
    for c in cols:
        if c in df.columns:
            vals = sorted(df[c].dropna().astype(str).unique())
            COLOR_MAPS[c] = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(vals)}


def cmap(col):
    return COLOR_MAPS.get(col) or None


SELECTABLE = ("bar", "stacked_bar", "barh", "line", "scatter", "pie", "donut", "treemap")
DONUT_ROOT = "Total"
DIM = "#D5DAE6"


def plotly_figure(df: pd.DataFrame, spec: dict, height=380, highlight: dict | None = None):
    t, x, y, color = spec["type"], spec.get("x"), spec.get("y"), spec.get("color")
    data = aggregate(df, spec)
    vl = value_label(spec)
    hl = {k: set(map(str, v)) for k, v in (highlight or {}).items()}
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sem dados para os filtros atuais", showarrow=False)
        return _layout(fig, spec, height)

    c = color if color and color != x and t not in ("pie", "donut") else None
    keys = [x] + ([c] if c else [])
    if t in ("bar", "stacked_bar"):
        fig = px.bar(data, x=x, y="valor", color=c, labels={"valor": vl}, custom_data=keys,
                     color_discrete_sequence=PALETTE, color_discrete_map=cmap(c), text_auto=True if not c else False)
        if not c:
            fig.update_traces(marker_color=BRAND["blue"])
    elif t == "barh":
        d2 = data.iloc[::-1] if not c else data
        fig = px.bar(d2, y=x, x="valor", color=c, orientation="h", labels={"valor": vl}, custom_data=keys,
                     color_discrete_sequence=PALETTE, color_discrete_map=cmap(c), text_auto=True if not c else False)
        if not c:
            fig.update_traces(marker_color=BRAND["blue"])
        fig.update_yaxes(categoryorder="array", categoryarray=list(data[x].cat.categories)[::-1])
    elif t in ("pie", "donut"):
        d2 = data.groupby(x, observed=True)["valor"].sum().reset_index()
        d2 = d2[d2["valor"] > 0]
        labels = [str(v) for v in d2[x]]
        cm = cmap(x) or {}
        cols = [cm.get(lab, PALETTE[i % len(PALETTE)]) for i, lab in enumerate(labels)]
        if hl.get(x):
            cols = [col if lab in hl[x] else DIM for lab, col in zip(labels, cols)]
        vals = list(d2["valor"])
        if t == "donut":
            tot = float(sum(vals))
            fig = go.Figure(go.Sunburst(
                ids=[DONUT_ROOT] + labels, labels=[f"{DONUT_ROOT}<br>{tot:,.1f}".replace(",", ".")] + labels,
                parents=[""] + [DONUT_ROOT] * len(labels), values=[tot] + vals, branchvalues="total",
                marker=dict(colors=["#FFFFFF"] + cols, line=dict(color="white", width=2)),
                textinfo="label+percent root", insidetextorientation="radial", maxdepth=2,
                hovertemplate="%{id}<br>%{value:,.2f} (%{percentRoot:.1%})<extra></extra>"))
        else:
            fig = go.Figure(go.Sunburst(
                ids=labels, labels=labels, parents=[""] * len(labels), values=vals,
                marker=dict(colors=cols, line=dict(color="white", width=2)), textinfo="label+percent root",
                insidetextorientation="radial",
                hovertemplate="%{label}<br>%{value:,.2f} (%{percentRoot:.1%})<extra></extra>"))
    elif t == "treemap":
        path = [x] + ([color] if color and color != x else [])
        d2 = data[data["valor"] > 0]
        fig = px.treemap(d2, path=path, values="valor", color=x, color_discrete_map=cmap(x),
                         color_discrete_sequence=PALETTE)
    elif t == "line":
        fig = px.line(data, x=x, y="valor", color=c, markers=True, labels={"valor": vl},
                      custom_data=keys, color_discrete_sequence=PALETTE, color_discrete_map=cmap(c))
    elif t == "histogram":
        fig = px.histogram(data, x=x, color=color, nbins=30, color_discrete_sequence=PALETTE)
    elif t == "scatter":
        fig = px.scatter(data, x=x, y=y, color=color, hover_data=["Nome", "Classe"],
                         custom_data=["GlobalId"], color_discrete_sequence=PALETTE, color_discrete_map=cmap(color))
        keys = ["GlobalId"]
    elif t == "box":
        fig = px.box(data, x=x, y=y, color=color, color_discrete_sequence=PALETTE)
    elif t == "table":
        fig = go.Figure(go.Table(
            header=dict(values=list(data.columns), fill_color=BRAND["navy"], font=dict(color="white")),
            cells=dict(values=[data[cn].astype(str) for cn in data.columns], fill_color="#F2F4FA")))
    elif t == "kpi":
        fig = go.Figure(go.Indicator(mode="number", value=float(data["valor"].iloc[0]),
                                     number=dict(font=dict(color=BRAND["blue"]))))
    else:
        fig = go.Figure()

    if hl and t in ("bar", "stacked_bar", "barh", "line", "scatter"):
        idx = {k: i for i, k in enumerate(keys)}
        for tr in fig.data:
            cd = getattr(tr, "customdata", None)
            if cd is None:
                continue
            ops = [1.0 if all(k not in idx or str(row[idx[k]]) in v for k, v in hl.items()) else 0.22
                   for row in cd]
            tr.marker.opacity = ops
    return _layout(fig, spec, height)


def points_to_conds(spec: dict, points: list[dict]) -> dict:
    conds: dict[str, list] = {}
    x = spec.get("x")
    c = spec.get("color") if spec.get("color") and spec.get("color") != x else None
    t = spec["type"]

    def add(col, val):
        if col and val is not None and str(val) not in conds.setdefault(col, []):
            conds[col].append(str(val))

    for p in points or []:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)) and cd:
            if t == "scatter":
                add("GlobalId", cd[0])
            else:
                add(x, cd[0])
                if c and len(cd) > 1:
                    add(c, cd[1])
        elif t in ("pie", "donut"):
            lab = p.get("id") or p.get("label")
            if lab and lab != DONUT_ROOT:
                add(x, lab)
        elif t == "treemap":
            parts = [s for s in str(p.get("id") or p.get("label") or "").split("/") if s]
            if parts:
                add(x, parts[0])
                if c and len(parts) > 1:
                    add(c, parts[1])
        else:
            add(x, p.get("y") if t == "barh" else p.get("x"))
    return {k: v for k, v in conds.items() if v}


MAX_TRIS = 600_000


def ifc_geometry(data: bytes, max_tris: int = MAX_TRIS):
    import multiprocessing
    import ifcopenshell.geom
    model = open_ifc_bytes(data)
    s = ifcopenshell.geom.settings()
    try:
        s.set("use-world-coords", True)
    except Exception:
        s.set(s.USE_WORLD_COORDS, True)
    it = ifcopenshell.geom.iterator(s, model, max(1, multiprocessing.cpu_count()))
    geo, tot, truncated = {}, 0, False
    if it.initialize():
        while True:
            sh = it.get()
            g = sh.geometry
            f = np.asarray(g.faces, dtype=np.int32).reshape(-1, 3)
            if tot + len(f) > max_tris:
                truncated = True
            elif len(f):
                geo[sh.guid] = (np.asarray(g.verts, dtype=np.float32).reshape(-1, 3), f)
                tot += len(f)
            if not it.next():
                break
    return geo, truncated


def viewer_figure(geo_maps: dict, df_vis: pd.DataFrame, df_ctx: pd.DataFrame | None,
                  color_by: str, height: int = 560):
    def collect(sub, with_text):
        V, F, T, off = [], [], [], 0
        for mdl, gid, nome, tipo in sub[["Modelo", "GlobalId", "Nome", "Tipo"]].itertuples(index=False):
            g = geo_maps.get(mdl, {}).get(gid)
            if g is None:
                continue
            v, f = g
            V.append(v); F.append(f + off); off += len(v)
            if with_text:
                T.append(np.full(len(v), f"{nome} · {tipo}", dtype=object))
        if not V:
            return None
        return np.vstack(V), np.vstack(F), (np.concatenate(T) if T else None)

    n_verts = sum(len(geo_maps.get(m, {}).get(g, (np.empty((0, 3)),))[0])
                  for m, g in zip(df_vis["Modelo"], df_vis["GlobalId"]))
    with_text = n_verts < 150_000
    traces, shown = [], 0
    for i, (val, sub) in enumerate(df_vis.groupby(color_by, sort=True)):
        r = collect(sub, with_text)
        if r is None:
            continue
        v, f, txt = r
        shown += sub["GlobalId"].isin(set().union(*[set(geo_maps.get(m, {})) for m in sub["Modelo"].unique()])).sum()
        traces.append(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=(cmap(color_by) or {}).get(str(val), PALETTE[i % len(PALETTE)]), name=str(val)[:40], showlegend=True, flatshading=True,
            lighting=dict(ambient=0.55, diffuse=0.75, specular=0.1, roughness=0.7, fresnel=0.1),
            lightposition=dict(x=2000, y=1000, z=5000),
            text=txt, hovertemplate=(f"<b>{str(val)[:40]}</b><br>%{{text}}<extra></extra>" if txt is not None
                                     else f"<b>{str(val)[:40]}</b><extra></extra>")))
    if df_ctx is not None and len(df_ctx):
        r = collect(df_ctx, False)
        if r is not None:
            v, f, _ = r
            traces.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                    color="#9AA3B5", opacity=0.08, name="Contexto (fora do filtro)",
                                    showlegend=True, hoverinfo="skip", flatshading=True))
    fig = go.Figure(traces)
    ax = dict(visible=False, showbackground=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), uirevision="ifc3d",
                      scene=dict(aspectmode="data", xaxis=ax, yaxis=ax, zaxis=ax, bgcolor="rgba(0,0,0,0)",
                                 camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))),
                      legend=dict(orientation="h", y=0.02, x=0.01, bgcolor="rgba(255,255,255,0.7)",
                                  font=dict(size=11)),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig, int(shown), n_verts


def mpl_png(df: pd.DataFrame, spec: dict, width=7.2, height=3.6, dpi=160) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, x, y, color = spec["type"], spec.get("x"), spec.get("y"), spec.get("color")
    data = aggregate(df, spec)
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_title(spec["title"], loc="left", fontsize=11, color=BRAND["navy"], fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)

    if data.empty:
        ax.text(0.5, 0.5, "Sem dados", ha="center", va="center"); ax.axis("off")
    elif t in ("bar", "stacked_bar", "barh", "line", "treemap"):
        if color and color != x and t != "treemap":
            pv = data.pivot_table(index=x, columns=color, values="valor", aggfunc="sum",
                                  observed=False).fillna(0)
            if t == "line":
                pv.plot(ax=ax, marker="o", color=PALETTE)
            else:
                pv.plot(kind="barh" if t == "barh" else "bar", stacked=True, ax=ax,
                        color=PALETTE[:len(pv.columns)] * 3, width=0.75)
            ax.legend(fontsize=7, frameon=False, ncol=min(len(pv.columns), 6),
                      loc="upper center", bbox_to_anchor=(0.5, -0.28))
        else:
            s = data.groupby(x, observed=True)["valor"].sum()
            labels = [str(i)[:28] for i in s.index]
            if t in ("barh", "treemap"):
                ax.barh(labels[::-1], s.values[::-1], color=BRAND["blue"])
                for i, v in enumerate(s.values[::-1]):
                    ax.text(v, i, f" {v:,.2f}".rstrip("0").rstrip(".").replace(",", "."), va="center", fontsize=7)
            elif t == "line":
                ax.plot(labels, s.values, marker="o", color=BRAND["blue"])
            else:
                ax.bar(labels, s.values, color=BRAND["blue"])
        ax.set_ylabel("" if t in ("barh", "treemap") else value_label(spec), fontsize=8)
        ax.set_xlabel("")
        if t not in ("barh", "treemap"):
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    elif t in ("pie", "donut"):
        s = data.groupby(x, observed=True)["valor"].sum()
        s = s[s > 0]
        ax.pie(s.values, labels=[str(i)[:22] for i in s.index], autopct="%1.0f%%",
               colors=PALETTE[:len(s)] * 3, textprops={"fontsize": 7},
               wedgeprops={"width": 0.45} if t == "donut" else None)
        ax.axis("equal")
    elif t == "histogram":
        ax.hist(pd.to_numeric(data[x], errors="coerce").dropna(), bins=30, color=BRAND["blue"])
        ax.set_xlabel(x, fontsize=8)
    elif t in ("scatter", "box"):
        if t == "scatter":
            ax.scatter(pd.to_numeric(data[x], errors="coerce"), pd.to_numeric(data[y], errors="coerce"),
                       s=10, color=BRAND["blue"], alpha=0.7)
        else:
            groups = [g[y].dropna().values for _, g in data.groupby(x)]
            ax.boxplot(groups, tick_labels=[str(k)[:15] for k, _ in data.groupby(x)])
        ax.set_xlabel(x, fontsize=8); ax.set_ylabel(y or "", fontsize=8)
    elif t == "kpi":
        ax.axis("off")
        ax.text(0.5, 0.45, f"{float(data['valor'].iloc[0]):,.2f}".replace(",", "."),
                ha="center", va="center", fontsize=28, color=BRAND["blue"], fontweight="bold")
    elif t == "table":
        ax.axis("off")
        d = data.head(15)
        tb = ax.table(cellText=d.astype(str).values, colLabels=list(d.columns), loc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 1.2)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def template_specs(df: pd.DataFrame) -> list[dict]:
    has_vol = df["Volume_m3"].notna().any()
    has_area = df["Area_m2"].notna().any()
    specs = [
        {"id": "T1", "title": "Elementos por classe IFC", "type": "barh", "x": "Classe", "agg": "count", "top_n": 15},
        {"id": "T2", "title": "Elementos por pavimento e classe", "type": "stacked_bar", "x": "Pavimento",
         "color": "Classe", "agg": "count", "sort": "x"},
        {"id": "T3", "title": "Volume (m³) por material" if has_vol else "Elementos por material",
         "type": "donut", "x": "Material", "y": "Volume_m3" if has_vol else None,
         "agg": "sum" if has_vol else "count", "top_n": 10,
         "filters": [{"column": "Classe", "op": "!=", "value": "IfcSpace"}]},
        {"id": "T4", "title": "Hierarquia Classe → Tipo", "type": "treemap", "x": "Classe", "color": "Tipo", "agg": "count"},
        {"id": "T5", "title": "Área (m²) por pavimento e classe" if has_area else "Tipos distintos por pavimento",
         "type": "stacked_bar", "x": "Pavimento", "y": "Area_m2" if has_area else "Tipo",
         "color": "Classe" if has_area else None,
         "agg": "sum" if has_area else "nunique", "sort": "x"},
        {"id": "T6", "title": "Quantitativo por classe", "type": "table", "x": "Classe", "agg": "count"},
    ]
    return [validate_spec(s, df)[0] for s in specs]


# ============================================================================
# ASSISTENTE DE IA (LLM)
# ============================================================================

PROVIDERS = {
    "Anthropic (Claude)": {"model": "claude-sonnet-3-5", "key_hint": "sk-ant-..."},
    "OpenAI": {"model": "gpt-4o-mini", "key_hint": "sk-..."},
    "Google Gemini": {"model": "gemini-2.5-flash", "key_hint": "AIza..."},
    "Compatível OpenAI (URL própria)": {"model": "llama-3.3-70b-versatile", "key_hint": "chave do provedor"},
}
FILTER_COLUMNS = ["Modelo", "Pavimento", "Classe", "Tipo", "Material"]


def data_context(df: pd.DataFrame, charts: list[dict], filters: dict) -> str:
    lines = [f"Total de elementos (após filtros): {len(df)}", "", "COLUNAS:"]
    for c in df.columns:
        if c == "GlobalId":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            lines.append(f"- {c} (numérica): n={len(s)}, soma={s.sum():.2f}, média={s.mean() if len(s) else 0:.2f}")
        else:
            vc = df[c].astype(str).value_counts()
            top = ", ".join(f"{k} ({v})" for k, v in vc.head(12).items())
            more = f" … +{len(vc) - 12} valores" if len(vc) > 12 else ""
            lines.append(f"- {c} (categórica, {len(vc)} valores): {top}{more}")
    lines += ["", "QUANTITATIVO POR CLASSE:"]
    q = df.groupby("Classe").agg(n=("GlobalId", "size"), vol=("Volume_m3", "sum"), area=("Area_m2", "sum"))
    lines += [f"- {i}: {r.n} un | {r.vol:.2f} m³ | {r.area:.2f} m²" for i, r in q.iterrows()]
    lines += ["", "GRÁFICOS NO PAINEL:"]
    lines += [f"- {c['id']}: {c['title']} [{c['type']}, x={c.get('x')}, y={c.get('y')}, agg={c.get('agg')}, "
              f"color={c.get('color')}]" for c in charts] or ["(nenhum)"]
    lines += ["", f"FILTROS GLOBAIS ATIVOS: {json.dumps(filters, ensure_ascii=False) if filters else 'nenhum'}"]
    return "\n".join(lines)


SYSTEM_PROMPT = """Você é o assistente do painel Especialização BIM UFPE - Do BIM à Inteligência Artificial,
especialista em BIM/IFC e visualização de dados. O usuário conversa em português sobre os
dados extraídos de modelos IFC e pede gráficos, filtros ou análises.

RESPONDA SEMPRE com UM ÚNICO objeto JSON válido (sem texto fora do JSON, sem markdown):
{
  "reply": "resposta curta em português, explicando o que fez ou respondendo à pergunta",
  "actions": [ ... zero ou mais ações ... ]
}

AÇÕES DISPONÍVEIS:
1. {"action": "add_chart", "spec": SPEC}
2. {"action": "update_chart", "id": "C1", "spec": {campos a alterar}}
3. {"action": "remove_chart", "id": "T3"}          (ids T* = template, C* = criados no chat)
4. {"action": "clear_user_charts"}                 (remove todos os C*)
5. {"action": "reset_template"}                    (restaura os gráficos T* originais)
6. {"action": "set_filter", "column": COL, "values": [valores exatos]}  COL em """ + str(FILTER_COLUMNS) + """
7. {"action": "reset_filters"}
8. {"action": "clear_selection"}                   (limpa a seleção cruzada feita com cliques nos gráficos)

SPEC de gráfico:
{"title": str, "type": um de """ + str(CHART_TYPES) + """,
 "x": coluna (categoria no eixo / fatias),
 "y": coluna numérica a agregar (omitir para contagem),
 "agg": um de """ + str(AGGS) + """,
 "color": coluna para empilhar/subdividir (opcional),
 "top_n": inteiro (opcional, 0 = todos),
 "filters": [{"column": COL, "op": um de """ + str(OPS) + """, "value": valor ou lista}] (filtro só deste gráfico)}

REGRAS:
- Use somente nomes de colunas e valores que existem no CONTEXTO DE DADOS.
- Volume_m3, Area_m2, Comprimento_m são quantidades; prefira agg "sum" para elas.
- "histogram" usa x numérico; "scatter"/"box" usam y numérico; "kpi" mostra um único número;
  "table" gera quantitativo agrupado por x (contagem + somas de volume/área/comprimento).
- Pavimento é ordenado por elevação automaticamente.
- Para perguntas sobre os dados, responda em "reply" com base no contexto (pode também criar um gráfico).
- Se o pedido for ambíguo, faça a melhor escolha e explique em "reply".
"""


def _post(url, headers, payload, timeout=90):
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def call_llm(provider: str, api_key: str, model: str, system: str,
             messages: list[dict], base_url: str | None = None) -> str:
    if provider.startswith("Anthropic"):
        js = _post("https://api.anthropic.com/v1/messages",
                   {"x-api-key": api_key, "anthropic-version": "2023-06-01",
                    "content-type": "application/json"},
                   {"model": model, "max_tokens": 2048, "system": system, "messages": messages})
        return "".join(b.get("text", "") for b in js.get("content", []) if b.get("type") == "text")

    if provider.startswith("Google"):
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]} for m in messages]
        js = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                   {"x-goog-api-key": api_key, "content-type": "application/json"},
                   {"system_instruction": {"parts": [{"text": system}]}, "contents": contents,
                    "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}})
        return js["candidates"][0]["content"]["parts"][0]["text"]

    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    payload = {"model": model, "temperature": 0.2,
               "messages": [{"role": "system", "content": system}] + messages}
    if provider == "OpenAI":
        payload["response_format"] = {"type": "json_object"}
    js = _post(url, {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}, payload)
    return js["choices"][0]["message"]["content"]


def parse_response(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        try:
            obj = json.loads(t[i:j + 1])
            if isinstance(obj, dict):
                obj.setdefault("reply", "")
                obj["actions"] = [a for a in obj.get("actions") or [] if isinstance(a, dict)]
                return obj
        except json.JSONDecodeError:
            pass
    return {"reply": text, "actions": []}


def ask(provider, api_key, model, base_url, df, charts, filters, history, question) -> dict:
    system = SYSTEM_PROMPT + "\n\nCONTEXTO DE DADOS:\n" + data_context(df, charts, filters)
    msgs = [{"role": m["role"], "content": m["raw"] if m["role"] == "assistant" else m["content"]}
            for m in history[-10:] if m["role"] in ("user", "assistant")]
    msgs.append({"role": "user", "content": question})
    raw = call_llm(provider, api_key, model, system, msgs, base_url)
    out = parse_response(raw)
    out["raw"] = raw
    return out


# ============================================================================
# RELATÓRIOS (HTML + PDF)
# ============================================================================

APP = {
    "name": "Especialização BIM UFPE",
    "sub": "Do BIM à Inteligência Artificial: a transformação digital disruptiva na indústria da Construção",
    "institution": "Célula BIM UFPE",
    "professor": "Ítalo Guedes",
}
HTML_NAME = "DataViz_BIM_UFPE_dashboard.html"


def fmt(v, dec=2):
    try:
        return f"{float(v):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return str(v)


def compute_kpis(df: pd.DataFrame) -> list[tuple[str, str]]:
    pav = df.loc[df["Pavimento"] != "(sem pavimento)", "Pavimento"].nunique()
    with_mat = (df["Material"] != "(sem material)").mean() * 100 if len(df) else 0
    return [
        ("Elementos", fmt(len(df), 0)),
        ("Classes IFC", fmt(df["Classe"].nunique(), 0)),
        ("Tipos", fmt(df["Tipo"].nunique(), 0)),
        ("Pavimentos", fmt(pav, 0)),
        ("Volume total (m³)", fmt(df["Volume_m3"].sum())),
        ("Área total (m²)", fmt(df["Area_m2"].sum())),
        ("Com material (%)", fmt(with_mat, 1)),
    ]


def build_html(df, charts, kpis, meta: dict, logo_bytes: bytes | None, fig3d=None) -> bytes:
    logo = f'<img src="data:image/png;base64,{base64.b64encode(logo_bytes).decode()}" alt="Célula BIM UFPE">' if logo_bytes else ""
    cards = "".join(f'<div class="kpi"><span>{html.escape(k)}</span><b>{v}</b></div>' for k, v in kpis)
    figs = []
    if fig3d is not None:
        fig3d.update_layout(height=560)
        figs.append('<div class="card wide"><h3 style="margin:6px 10px;color:#322F44">Modelo 3D</h3>'
                    + fig3d.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False})
                    + "</div>")
    for i, spec in enumerate(charts):
        fig = plotly_figure(df, spec, height=420 if spec["type"] != "table" else 460)
        figs.append(f'<div class="card">{fig.to_html(full_html=False, include_plotlyjs=(i == 0 and fig3d is None), config={"displaylogo": False})}</div>')
    filt = html.escape(meta.get("filters") or "Nenhum")
    models = html.escape(", ".join(meta.get("models", [])))
    notes = f'<p class="notes">{html.escape(meta["notes"])}</p>' if meta.get("notes") else ""
    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{APP['name']} — Painel</title>
<style>
:root{{--blue:{BRAND['blue']};--green:{BRAND['green']};--navy:{BRAND['navy']}}}
body{{margin:0;font-family:Inter,'Segoe UI',Arial,sans-serif;background:#F6F8FC;color:#2b2940}}
header{{background:#fff;border-bottom:4px solid var(--green);padding:18px 28px;display:flex;gap:24px;align-items:center;flex-wrap:wrap}}
header img{{height:54px}} header h1{{margin:0;font-size:22px;color:var(--navy)}}
header p{{margin:4px 0 0;color:#5b5a70;font-size:13px}}
main{{max-width:1400px;margin:auto;padding:20px 16px}}
.meta{{font-size:13px;color:#4a4960;margin-bottom:14px}} .meta b{{color:var(--navy)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}}
.kpi{{background:#fff;border-radius:10px;padding:12px 14px;border-left:4px solid var(--blue);box-shadow:0 1px 3px #0001}}
.kpi span{{display:block;font-size:12px;color:#6b6a80}} .kpi b{{font-size:22px;color:var(--navy)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px}}
.card{{background:#fff;border-radius:10px;padding:8px;box-shadow:0 1px 3px #0001;overflow:hidden}}
.wide{{grid-column:1/-1}}
.notes{{background:#fff;border-radius:10px;padding:12px;border-left:4px solid var(--green)}}
footer{{text-align:center;font-size:12px;color:#777;padding:20px}}
@media(max-width:600px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header>{logo}<div><h1>{APP['name']}</h1>
<p>{APP['sub']}<br/>{APP['institution']} · Professor: {APP['professor']}</p></div></header>
<main>
<div class="meta"><b>Relatório:</b> {html.escape(meta.get('title', 'Painel de dados IFC'))} &nbsp;|&nbsp;
<b>Modelos:</b> {models} &nbsp;|&nbsp; <b>Filtros:</b> {filt} &nbsp;|&nbsp;
<b>Gerado em:</b> {meta.get('date')}</div>
{notes}
<div class="kpis">{cards}</div>
<div class="grid">{''.join(figs)}</div>
</main><footer>{APP['name']} · {APP['institution']} · Prof. {APP['professor']}</footer>
</body></html>"""
    return page.encode("utf-8")


def build_pdf(df, charts, kpis, meta: dict, logo_bytes: bytes | None, html_bytes: bytes | None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    font, bold = _register_fonts()
    navy, blue, green = (colors.HexColor(BRAND[k]) for k in ("navy", "blue", "green"))
    ss = getSampleStyleSheet()
    for k in ("Title", "Heading2", "BodyText"):
        ss[k].fontName = bold if k != "BodyText" else font
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=navy, fontSize=18, alignment=TA_LEFT, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=navy, fontSize=12, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))
    box = ParagraphStyle("box", parent=body, backColor=colors.HexColor("#EAF9F2"), borderColor=green,
                         borderWidth=1, borderPadding=7, spaceBefore=6, spaceAfter=10)

    def on_page(c, doc):
        w, h = A4
        c.saveState()
        if logo_bytes:
            from reportlab.lib.utils import ImageReader
            c.drawImage(ImageReader(io.BytesIO(logo_bytes)), 1.6 * cm, h - 1.75 * cm, width=4.2 * cm,
                        height=2.2 * cm, preserveAspectRatio=True, mask="auto")
        c.setFont(bold, 9); c.setFillColor(navy)
        c.drawRightString(w - 1.6 * cm, h - 1.05 * cm, APP["name"])
        c.setFont(font, 8); c.setFillColor(colors.grey)
        c.drawRightString(w - 1.6 * cm, h - 1.45 * cm, f"{APP['institution']} · Prof. {APP['professor']}")
        c.setStrokeColor(green); c.setLineWidth(2); c.line(1.6 * cm, h - 1.9 * cm, w - 1.6 * cm, h - 1.9 * cm)
        c.setFont(font, 7.5); c.setFillColor(colors.grey)
        c.drawString(1.6 * cm, 1.0 * cm, f"Gerado em {meta.get('date')}")
        c.drawRightString(w - 1.6 * cm, 1.0 * cm, f"Página {doc.page}")
        c.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=2.4 * cm, bottomMargin=1.6 * cm,
                            title=f"{APP['name']} — {meta.get('title', '')}", author=APP["professor"],
                            subject="Relatório de dados IFC")
    W = A4[0] - 3.2 * cm
    st = [Paragraph(html.escape(meta.get("title") or "Relatório do Painel IFC"), h1),
          Paragraph(f"<b>Aplicação:</b> {APP['name']} - {APP['sub']}<br/>"
                    f"<b>Instituição:</b> {APP['institution']} &nbsp; <b>Professor:</b> {APP['professor']}", body),
          Paragraph(f"<b>Modelos:</b> {html.escape(', '.join(meta.get('models', [])))}", body),
          Paragraph(f"<b>Filtros aplicados:</b> {html.escape(meta.get('filters') or 'Nenhum')}", body)]
    if meta.get("author"):
        st.append(Paragraph(f"<b>Elaborado por:</b> {html.escape(meta['author'])}", body))
    if html_bytes:
        st.append(Paragraph(
            f"<b>Painel interativo (.html):</b> este PDF contém o arquivo <b>{HTML_NAME}</b> anexado. "
            "Abra o painel de <b>Anexos</b> no seu leitor de PDF e clique para visualizar o dashboard no navegador.", box))
    if meta.get("notes"):
        st += [Paragraph("Observações", h2), Paragraph(html.escape(meta["notes"]).replace("\n", "<br/>"), body)]

    st.append(Paragraph("Indicadores-chave", h2))
    cells = [[Paragraph(f"<font size=7.5 color='#666666'>{k}</font><br/><font size=12><b>{v}</b></font>", body)
              for k, v in kpis[i:i + 4]] for i in range(0, len(kpis), 4)]
    cells[-1] += [""] * (4 - len(cells[-1]))
    t = Table(cells, colWidths=[W / 4] * 4)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F5FC")),
                           ("LINEBEFORE", (0, 0), (-1, -1), 3, blue),
                           ("INNERGRID", (0, 0), (-1, -1), 4, colors.white),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    st.append(t)

    st.append(Paragraph("Painel de gráficos", h2))
    for spec in charts:
        if spec["type"] == "table":
            continue
        png = mpl_png(df, spec)
        img = Image(io.BytesIO(png))
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth, img.drawHeight = W, W * ratio
        if img.drawHeight > 11 * cm:
            img.drawHeight, img.drawWidth = 11 * cm, 11 * cm / ratio
        st.append(KeepTogether([img, Paragraph(f"[{spec['id']}] {html.escape(spec['title'])}", small), Spacer(1, 8)]))

    tables = [s for s in charts if s["type"] == "table"] or [
        {"id": "Q", "title": "Quantitativo por classe", "type": "table", "x": "Classe", "agg": "count"}]
    for spec in tables:
        d = aggregate(df, spec).head(40)
        st.append(Paragraph(html.escape(spec["title"]), h2))
        data = [list(d.columns)] + [[fmt(v) if isinstance(v, float) else str(v) for v in r] for r in d.values]
        tb = Table(data, repeatRows=1, colWidths=[W * 0.4] + [W * 0.6 / max(len(d.columns) - 1, 1)] * (len(d.columns) - 1))
        tb.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 0), bold),
                                ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5FC")]),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DAE6"))]))
        st.append(tb)
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    pdf = buf.getvalue()

    if html_bytes:
        from pypdf import PdfReader, PdfWriter
        writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf)))
        writer.add_attachment(HTML_NAME, html_bytes)
        out = io.BytesIO(); writer.write(out)
        pdf = out.getvalue()
    return pdf


def _register_fonts():
    import os
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping
    d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    try:
        if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(d, "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(d, "DejaVuSans-Bold.ttf")))
            addMapping("DejaVu", 0, 0, "DejaVu"); addMapping("DejaVu", 1, 0, "DejaVu-Bold")
            addMapping("DejaVu", 0, 1, "DejaVu"); addMapping("DejaVu", 1, 1, "DejaVu-Bold")
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def now_str():
    return datetime.now().strftime("%d/%m/%Y %H:%M")


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

FILTER_COLS = FILTER_COLUMNS

st.set_page_config(page_title=APP["name"], page_icon="📊", layout="wide")

st.markdown(f"""
<style>
:root {{ --zblue:{BRAND['blue']}; --zgreen:{BRAND['green']}; --znavy:{BRAND['navy']}; }}
.block-container {{ padding-top: 1.4rem; }}
.zhead h1 {{ color: var(--znavy); font-size: 1.6rem; margin: 0; }}
.zhead p {{ color: #5b5a70; margin: .2rem 0 0; font-size: 0.95rem; }}
.zbar {{ height: 4px; background: linear-gradient(90deg,var(--zblue),var(--zgreen)); border-radius: 4px; margin: .6rem 0 1rem; }}
.kpi {{ background:#fff; border-radius:10px; padding:.7rem .9rem; border-left:4px solid var(--zblue);
        box-shadow:0 1px 3px rgba(0,0,0,.08); }}
.kpi span {{ font-size:.78rem; color:#6b6a80; display:block; }}
.kpi b {{ font-size:1.12rem; color:var(--znavy); white-space:nowrap; }}
.kpi {{ min-height: 4.6rem; }}
.cid {{ display:inline-block; background:var(--znavy); color:#fff; border-radius:6px; padding:0 .45rem;
        font-size:.72rem; font-weight:600; }}
</style>""", unsafe_allow_html=True)


def init_state():
    ss = st.session_state
    ss.setdefault("models", {})
    ss.setdefault("template", None)
    ss.setdefault("user_charts", [])
    ss.setdefault("hidden", set())
    ss.setdefault("chat", [])
    ss.setdefault("counter", 0)
    ss.setdefault("xsel", {})
    for c in FILTER_COLS:
        ss.setdefault(f"f_{c}", [])
    for c, vals in ss.pop("pending_filters", {}).items():
        ss[f"f_{c}"] = vals


init_state()
ss = st.session_state


def queue_filter(col, vals):
    ss.setdefault("pending_filters", {})[col] = vals


@st.cache_data(show_spinner=False, max_entries=10)
def load_model(digest: str, name: str, data: bytes, geom: bool):
    model = open_ifc_bytes(data)
    df, props = extract(model, name, geometric_quantities=geom)
    return df, props, model_info(model)


@st.cache_resource(show_spinner=False, max_entries=6)
def load_geometry(digest: str, _data: bytes):
    return ifc_geometry(_data)


def next_id():
    ss.counter += 1
    return f"C{ss.counter}"


# =================================================================== cabeçalho
c_logo, c_title = st.columns([1.2, 4], vertical_alignment="center")
if LOGO_BYTES:
    # use_container_width=True faz a logo redimensionar perfeitamente sem cortes
    c_logo.image(LOGO_BYTES, use_container_width=True)
c_title.markdown(f"""<div class="zhead"><h1>{APP['name']}</h1>
<p>{APP['sub']}<br/><b>{APP['institution']}</b> · Professor: {APP['professor']}</p></div>""",
                 unsafe_allow_html=True)
st.markdown('<div class="zbar"></div>', unsafe_allow_html=True)


# =================================================================== sidebar: modelos
with st.sidebar:
    st.header("1 · Modelos IFC")
    ups = st.file_uploader("Carregar arquivos .ifc", type=["ifc"], accept_multiple_files=True)
    for up in ups or []:
        ss.models[up.name] = up.getvalue()

    with st.expander("Carregar por URL"):
        url = st.text_input("URL direta de um arquivo .ifc", placeholder="https://.../modelo.ifc")
        if st.button("Baixar modelo", disabled=not url):
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                if not r.content.lstrip()[:20].startswith(b"ISO-10303-21"):
                    st.error("O conteúdo não parece ser um arquivo IFC (STEP).")
                else:
                    ss.models[url.rstrip("/").split("/")[-1].split("?")[0] or "modelo_url.ifc"] = r.content
                    st.success("Modelo carregado.")
            except Exception as e:
                st.error(f"Falha ao baixar: {e}")

    if st.button("🏢 Usar modelo de exemplo", width="stretch"):
        ss.models["exemplo_edificio.ifc"] = build_sample_ifc()

    geom = st.toggle("Calcular quantidades pela geometria quando faltar Qto (mais lento)", value=False)

    if ss.models:
        st.caption("Modelos na sessão:")
        for name in list(ss.models):
            a, b = st.columns([5, 1])
            size_mb = len(ss.models[name]) / 1e6
            a.write(f"• {name}<br/><small>{size_mb:.1f} MB</small>", unsafe_allow_html=True)
            if b.button("X", key=f"rm_{name}", help="Remover modelo"):
                ss.models.pop(name)
                st.rerun()

# =================================================================== carga de dados
if not ss.models:
    st.info("Carregue um ou mais modelos **.ifc** na barra lateral (ou use o modelo de exemplo) para montar o painel.")
    st.markdown("""
**O que este painel faz**
- Extrai elementos, classes, tipos, pavimentos, materiais, Psets e quantidades (Qto) dos modelos IFC;
- Monta um **template visual pré-definido** (KPIs + 6 visualizações) com filtros globais;
- Permite conversar com uma **LLM** (Claude, GPT, Gemini ou compatível) para criar/editar gráficos;
- Exporta **relatório PDF** com o painel e o **dashboard interativo .html anexado**.
    """)
    st.stop()

frames, props_frames, infos = [], [], {}
prog = st.progress(0.0, "Lendo modelos...")
for i, (name, data) in enumerate(ss.models.items()):
    digest = hashlib.md5(data).hexdigest()
    try:
        d, p, info = load_model(digest, name, data, geom)
        frames.append(d)
        props_frames.append(p)
        infos[name] = info
    except Exception as e:
        st.error(f"Não foi possível ler **{name}**: {e}")
    prog.progress((i + 1) / len(ss.models), f"Modelo {i + 1}/{len(ss.models)} processado")
prog.empty()

if not frames:
    st.stop()

DF = pd.concat(frames, ignore_index=True)
PROPS = pd.concat(props_frames, ignore_index=True)

if ss.template is None:
    ss.template = template_specs(DF)

set_color_maps(DF, ["Classe", "Pavimento", "Material", "Tipo", "Modelo", "Externo", "Estrutural", "ResistenciaFogo", "Fonte_Quantidade"])

# =================================================================== sidebar: filtros + IA
with st.sidebar:
    st.header("2 · Filtros")
    base = DF
    for col in FILTER_COLS:
        opts = sorted(base[col].dropna().astype(str).unique())
        ss[f"f_{col}"] = [v for v in ss[f"f_{col}"] if v in opts]
        sel = st.multiselect(col, opts, key=f"f_{col}", placeholder="Todos")
        if sel:
            base = base[base[col].astype(str).isin(sel)]
    
    if st.button("Limpar filtros", width="stretch"):
        for col in FILTER_COLS:
            queue_filter(col, [])
        st.rerun()

    st.header("3 · Assistente IA")
    provider = st.selectbox("Provedor", list(PROVIDERS))
    api_key = st.text_input("Chave API", type="password", placeholder=PROVIDERS[provider]["key_hint"], help="A chave fica apenas na sessão do navegador; não é gravada em disco.")
    model_name = st.text_input("Modelo", value=PROVIDERS[provider]["model"], key=f"model_{provider}")
    
    base_url = None
    if provider.startswith("Compatível"):
        base_url = st.text_input("Base URL", value="https://api.groq.com/openai/v1", help="Ex.: https://openrouter.ai/api/v1")
    
    try:
        default_key = st.secrets.get("LLM_API_KEY", "")
    except Exception:
        default_key = ""
        
    api_key = api_key or default_key
    st.caption("🟢 IA ativa" if api_key else "🔴 Informe a chave para ativar o chat")

ACTIVE_FILTERS = {c: ss[f"f_{c}"] for c in FILTER_COLS if ss[f"f_{c}"]}
DFF = base

def all_charts():
    return [c for c in ss.template if c["id"] not in ss.hidden] + ss.user_charts

def xfilter(df: pd.DataFrame, exclude: str | None = None) -> pd.DataFrame:
    d = df
    for cid, conds in ss.xsel.items():
        if cid == exclude:
            continue
        for col, vals in conds.items():
            if col in d.columns:
                d = d[d[col].astype(str).isin(vals)]
    return d

def on_chart_select(cid: str):
    state = ss.get(f"pl_{cid}") or {}
    try:
        points = state["selection"]["points"]
    except (KeyError, TypeError):
        points = []
    spec = next((c for c in all_charts() if c["id"] == cid), None)
    conds = points_to_conds(spec, points) if spec else {}
    if conds:
        ss.xsel[cid] = conds
    else:
        ss.xsel.pop(cid, None)

def _short(vals, n=4):
    vals = list(vals)
    return ", ".join(vals[:n]) + (f" (+{len(vals)-n})" if len(vals) > n else "")

XSEL_TXT = " | ".join(f"{cid}: " + " & ".join(f"{col} ({_short(v)})" for col, v in conds.items()) for cid, conds in ss.xsel.items())
DFX = xfilter(DFF)
FILTER_TXT = "; ".join(f"{k}: {', '.join(v)}" for k, v in ACTIVE_FILTERS.items()) or "Nenhum"
if XSEL_TXT:
    FILTER_TXT += f" | Seleção por clique: {XSEL_TXT}"

def apply_actions(actions: list[dict]) -> list[str]:
    log = []
    for a in actions:
        kind = a.get("action")
        try:
            if kind == "add_chart":
                spec, w = validate_spec(a.get("spec") or {}, DF)
                spec["id"] = next_id()
                ss.user_charts.append(spec)
                log += [f"➕ {spec['id']}: {spec['title']}"] + [f"⚠️ {x}" for x in w]
            elif kind == "update_chart":
                cid = a.get("id")
                pool = ss.user_charts if str(cid).startswith("C") else ss.template
                for i, c in enumerate(pool):
                    if c["id"] == cid:
                        merged = {**c, **(a.get("spec") or {})}
                        if "title" not in (a.get("spec") or {}) and any(k in (a.get("spec") or {}) for k in ("x", "y", "agg", "type")):
                            merged["title"] = None
                        spec, w = validate_spec(merged, DF)
                        spec["id"] = cid
                        pool[i] = spec
                        ss.hidden.discard(cid)
                        log += [f"🔄 {cid} atualizado"] + [f"⚠️ {x}" for x in w]
                        break
                else:
                    log.append(f"❌ Gráfico {cid} não encontrado")
            elif kind == "remove_chart":
                cid = a.get("id")
                if str(cid).startswith("T"):
                    ss.hidden.add(cid)
                else:
                    ss.user_charts = [c for c in ss.user_charts if c["id"] != cid]
                log.append(f"🗑️ {cid} removido")
            elif kind == "clear_user_charts":
                ss.user_charts = []
                log.append("🗑️ Gráficos do chat removidos")
            elif kind == "reset_template":
                ss.template = template_specs(DF)
                ss.hidden = set()
                log.append("🔄 Template restaurado")
            elif kind == "set_filter":
                col = a.get("column")
                if col in FILTER_COLS:
                    vals = a.get("values") or []
                    valid = set(DF[col].astype(str).unique())
                    chosen = [str(v) for v in (vals if isinstance(vals, list) else [vals]) if str(v) in valid]
                    queue_filter(col, chosen)
                    log.append(f"🔍 Filtro {col}: {', '.join(chosen) or 'todos'}")
            elif kind == "clear_selection":
                ss.xsel = {}
                log.append("🧹 Seleção cruzada limpa")
            elif kind == "reset_filters":
                for col in FILTER_COLS:
                    queue_filter(col, [])
                log.append("🧹 Filtros limpos")
        except Exception as e:
            log.append(f"❌ Ação {kind} falhou: {e}")
    return log

# =================================================================== abas
tab_dash, tab_data, tab_rep, tab_help = st.tabs(["📊 Painel", "🗃️ Dados & Propriedades", "📄 Relatório", "❓ Ajuda"])

with tab_dash:
    show_chat = st.toggle("💬 Mostrar assistente IA ao lado do painel", value=True)
    col_dash, col_chat = st.columns([2.3, 1]) if show_chat else (st.container(), None)

    with col_dash:
        st.caption(f"**Modelos:** {', '.join(ss.models)} | **Filtros:** {FILTER_TXT}")
        kpis = compute_kpis(DFX)
        kc = st.columns(len(kpis))
        for c, (k, v) in zip(kc, kpis):
            c.markdown(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>', unsafe_allow_html=True)
        st.write("")

        if ss.xsel:
            s1, s2 = st.columns([5, 1], vertical_alignment="center")
            s1.info(f"🎯 **Seleção ativa** ({len(DFX)} elementos) — {XSEL_TXT}")
            if s2.button("X Limpar seleção", width="stretch"):
                ss.xsel = {}
                st.rerun()
        else:
            st.caption("💡 Clique em uma barra, fatia ou bloco para filtrar todo o painel e o modelo 3D (Shift+clique seleciona vários; clique duplo no gráfico desfaz).")

        with st.container(border=True):
            v1, v2, v3, v4 = st.columns([2.4, 1, 1.4, 1.2], vertical_alignment="center")
            v1.markdown("**🧊 Visualizador 3D do modelo IFC**")
            show3d = v2.toggle("Exibir", value=True, key="v_on")
            color_by = v3.selectbox("Colorir por", ["Classe", "Pavimento", "Material", "Tipo", "Modelo", "Externo", "Estrutural"], key="v_color", label_visibility="collapsed")
            ghost = v4.toggle("Contexto", value=True, key="v_ghost", help="Mostra em cinza transparente os elementos fora do filtro/seleção")

            if show3d:
                geo_maps, trunc = {}, False
                with st.spinner("Processando geometria 3D (só na primeira vez)..."):
                    for name, data in ss.models.items():
                        try:
                            g, t = load_geometry(hashlib.md5(data).hexdigest(), data)
                            geo_maps[name] = g
                            trunc = trunc or t
                        except Exception as e:
                            st.warning(f"Geometria de {name} indisponível: {e}")

                ctx = None
                if ghost and len(DFX) < len(DF):
                    ctx = DF[~DF.set_index(["Modelo", "GlobalId"]).index.isin(DFX.set_index(["Modelo", "GlobalId"]).index)]

                fig3d, n_show, n_verts = viewer_figure(geo_maps, DFX, ctx, color_by)
                ss.last_fig3d = fig3d if n_verts < 400_000 else None
                st.plotly_chart(fig3d, width="stretch", key="viewer3d", config={"displaylogo": False, "scrollZoom": True})
                n_geo = sum(len(g) for g in geo_maps.values())
                st.caption(f"{n_show} de {len(DFX)} elementos filtrados exibidos ({n_geo} elementos com geometria no(s) modelo(s)). Arraste para girar, role para zoom, botão direito para mover." + (" (Modelo grande: geometria limitada a 600 mil triângulos.)" if trunc else ""))
            else:
                ss.last_fig3d = None

        charts_now = all_charts()
        for row in range(0, len(charts_now), 2):
            cols = st.columns(2)
            for col, spec in zip(cols, charts_now[row: row + 2]):
                with col.container(border=True):
                    h1, h2 = st.columns([6, 1])
                    h1.markdown(f'<span class="cid">{spec["id"]}</span>', unsafe_allow_html=True)
                    if h2.button("🗑️", key=f"del_{spec['id']}", help="Remover do painel"):
                        apply_actions([{"action": "remove_chart", "id": spec["id"]}])
                        st.rerun()

                    dsrc = xfilter(DFF, exclude=spec["id"])
                    if spec["type"] == "table":
                        st.markdown(f"**{spec['title']}**")
                        st.dataframe(aggregate(dsrc, spec), hide_index=True, width="stretch", height=330)
                    elif spec["type"] in SELECTABLE:
                        st.plotly_chart(plotly_figure(dsrc, spec, highlight=ss.xsel.get(spec["id"])),
                                        width="stretch", config={"displaylogo": False}, key=f"pl_{spec['id']}",
                                        on_select=lambda cid=spec["id"]: on_chart_select(cid),
                                        selection_mode=("points", "box", "lasso") if spec["type"] == "scatter" else ("points",))
                    else:
                        st.plotly_chart(plotly_figure(dsrc, spec), width="stretch", config={"displaylogo": False}, key=f"pl_{spec['id']}")

        with st.expander("➕ Criar gráfico manualmente (sem IA)"):
            cats = [c for c in CATEGORICAL if c in DF.columns]
            nums = [c for c in NUMERIC if c in DF.columns]
            a, b, c, d = st.columns(4)
            ctype = a.selectbox("Tipo", CHART_TYPES, key="m_type")
            cx = b.selectbox("Eixo X / categoria", cats + nums, key="m_x")
            cagg = c.selectbox("Agregação", AGGS, key="m_agg")
            cy = d.selectbox("Valor (Y)", ["(contagem)"] + nums, key="m_y")
            e, f, g = st.columns([1, 1, 2])
            ccol = e.selectbox("Cor / subgrupo", ["(nenhum)"] + cats, key="m_color")
            ctop = f.number_input("Top N (0 = todos)", 0, 100, 0, key="m_top")
            ctitle = g.text_input("Título (opcional)", key="m_title")

            if st.button("Adicionar ao painel", type="primary"):
                apply_actions([{"action": "add_chart", "spec": {
                    "type": ctype, "x": cx, "agg": cagg if cy != "(contagem)" else "count",
                    "y": None if cy == "(contagem)" else cy,
                    "color": None if ccol == "(nenhum)" else ccol, "top_n": ctop, "title": ctitle or None}}])
                st.rerun()

        b1, b2 = st.columns(2)
        if b1.button("🔄 Restaurar template padrão", width="stretch"):
            apply_actions([{"action": "reset_template"}])
            st.rerun()
        if b2.button("🗑️ Remover gráficos criados", width="stretch"):
            apply_actions([{"action": "clear_user_charts"}])
            st.rerun()

    if col_chat is not None:
        with col_chat:
            st.subheader("✨ Assistente IA")
            box = st.container(height=560)
            with box:
                if not ss.chat:
                    st.markdown("Exemplos de pedidos:\n"
                                "- *Crie um gráfico de pizza do volume por material só das paredes*\n"
                                "- *Mostre a área por pavimento empilhada por classe*\n"
                                "- *Filtre apenas o Térreo e o Pavimento 1*\n"
                                "- *Remova o gráfico T4 e transforme o C1 em barras horizontais*\n"
                                "- *Qual classe tem o maior volume?*")
                for m in ss.chat:
                    with st.chat_message(m["role"], avatar="👤" if m["role"] == "user" else "🤖"):
                        st.markdown(m["content"])

            q = st.chat_input("Peça um gráfico, filtro ou análise...", disabled=not api_key)
            if q:
                ss.chat.append({"role": "user", "content": q})
                try:
                    with st.spinner("Pensando..."):
                        out = ask(provider, api_key, model_name, base_url, DFX, all_charts(),
                                  {**ACTIVE_FILTERS, **({"seleção_por_clique": XSEL_TXT} if XSEL_TXT else {})},
                                  ss.chat[:-1], q)
                        log = apply_actions(out["actions"])
                        text = out["reply"] or "Pronto."
                        if log:
                            text += "\n\n" + "\n".join(f"- {x}" for x in log)
                        ss.chat.append({"role": "assistant", "content": text, "raw": out["raw"]})
                except Exception as e:
                    err = f"❌ Erro ao chamar a LLM: {e}"
                    ss.chat.append({"role": "assistant", "content": err, "raw": json.dumps({"reply": err})})
                st.rerun()

            if ss.chat and st.button("🧹 Limpar conversa", width="stretch"):
                ss.chat = []
                st.rerun()

with tab_data:
    st.subheader("Modelos carregados")
    st.dataframe(pd.DataFrame(infos).T, width="stretch")

    st.subheader(f"Elementos filtrados ({len(DFX)})")
    st.dataframe(DFX, width="stretch", height=380, hide_index=True)
    st.download_button("⬇️ Baixar CSV dos elementos", DFX.to_csv(index=False, sep=";").encode("utf-8-sig"), "elementos_ifc.csv", "text/csv")

    st.subheader("Explorador de propriedades (Psets / Qtos)")
    pf = PROPS[PROPS["GlobalId"].isin(DFX["GlobalId"])]
    if pf.empty:
        st.info("Nenhuma propriedade para os elementos filtrados.")
    else:
        a, b = st.columns(2)
        pset = a.selectbox("Pset / Qto", sorted(pf["Pset"].unique()))
        prop = b.selectbox("Propriedade", sorted(pf.loc[pf["Pset"] == pset, "Propriedade"].unique()))
        sel = pf[(pf["Pset"] == pset) & (pf["Propriedade"] == prop)]
        vc = sel["Valor"].replace("", "(vazio)").value_counts().reset_index()
        vc.columns = ["Valor", "Elementos"]
        c1, c2 = st.columns([1, 1.4])
        c1.dataframe(vc, hide_index=True, width="stretch", height=300)

        top = vc.head(20).iloc[::-1]
        fig = px.bar(top, x="Elementos", y=top["Valor"].astype(str).str[:40], orientation="h", title=f"{pset}.{prop}", template="plotly_white", height=320)
        fig.update_traces(marker_color=BRAND["blue"]).update_layout(yaxis_title="", margin=dict(t=40, l=10, r=10, b=10))
        c2.plotly_chart(fig, width="stretch", key="prop_fig")

    st.subheader("Completude de dados por classe (% elementos com o Pset)")
    top_psets = pf["Pset"].value_counts().head(12).index
    m = pf[pf["Pset"].isin(top_psets)][["GlobalId", "Pset"]].drop_duplicates().merge(DFX[["GlobalId", "Classe"]], on="GlobalId")
    tot = DFX.groupby("Classe").size()
    heat = (m.groupby(["Classe", "Pset"]).size().unstack(fill_value=0).div(tot, axis=0) * 100).round(0)
    st.dataframe(heat.style.background_gradient(cmap="Blues", vmin=0, vmax=100).format("{:.0f} %"), width="stretch")

with tab_rep:
    st.subheader("Relatório do painel")
    a, b = st.columns(2)
    title = a.text_input("Título do relatório", value="Painel de dados IFC - Especialização BIM UFPE")
    author = b.text_input("Elaborado por (aluno/equipe)", value="")
    notes = st.text_area("Observações / análise (aparece no PDF e no HTML)", height=100)
    attach = st.checkbox(f"Anexar o dashboard interativo ({HTML_NAME}) dentro do PDF", value=True)

    st.caption(f"Serão exportados {len(all_charts())} gráficos com os filtros atuais: {FILTER_TXT}")
    if st.button("📄 Gerar relatório", type="primary"):
        meta = {"title": title, "author": author, "notes": notes, "models": list(ss.models), "filters": FILTER_TXT, "date": now_str()}
        with st.spinner("Gerando HTML e PDF..."):
            charts_now = all_charts()
            kp = compute_kpis(DFX)
            html_bytes = build_html(DFX, charts_now, kp, meta, LOGO_BYTES, fig3d=ss.get("last_fig3d"))
            pdf_bytes = build_pdf(DFX, charts_now, kp, meta, LOGO_BYTES, html_bytes if attach else None)
            ss.report = (pdf_bytes, html_bytes)
        st.success("Relatório gerado!")

    if ss.get("report"):
        pdf_bytes, html_bytes = ss.report
        c1, c2 = st.columns(2)
        c1.download_button("📄 Baixar relatório PDF", pdf_bytes, "DataViz_BIM_UFPE_relatorio.pdf", "application/pdf", width="stretch", type="primary")
        c2.download_button("🌐 Baixar dashboard.html", html_bytes, HTML_NAME, "text/html", width="stretch")
        st.info("💡 **O PDF traz o dashboard .html como anexo** (ícone de clipe no Acrobat Reader, Firefox ou Foxit).")

        with st.expander("👁️ Pré-visualizar o dashboard.html"):
            if hasattr(st, "iframe"):
                st.iframe(html_bytes.decode("utf-8"), height=900)
            else:
                import streamlit.components.v1 as components
                components.html(html_bytes.decode("utf-8"), height=900, scrolling=True)

with tab_help:
    st.markdown(f"""
### Como usar
1. **Modelos IFC:** carregue um ou vários `.ifc` (IFC2x3 / IFC4 / IFC4x3). Cada modelo vira a coluna "Modelo", permitindo comparar disciplinas ou revisões.
2. **Filtros:** Modelo, Pavimento, Classe, Tipo e Material filtram todo o painel, o chat e o relatório.
3. **Template:** gráficos T1-T6 são o padrão visual; os criados no chat/manual recebem C1, C2...
4. **Assistente IA:** escolha o provedor, cole a chave API e converse. A IA devolve apenas especificações JSON (nunca código), que o app valida antes de aplicar.
5. **Filtragem cruzada:** clique numa barra, fatia ou bloco de qualquer gráfico: KPIs, demais gráficos, modelo 3D, tabela de dados e relatório passam a mostrar só aquela seleção. Shift+clique soma itens; "Limpar seleção" desfaz.
6. **Visualizador 3D:** geometria triangulada do IFC, colorida por Classe, Pavimento, Material; segue filtros e seleções.
7. **Relatório:** gera PDF (KPIs, gráficos, quantitativos) com o dashboard interativo html anexado.

### Colunas extraídas
`{', '.join(DF.columns)}`

**Quantidades:** vêm dos conjuntos `Qto_` do IFC (exporte o Revit com "Exportar quantidades base" ativado). Sem `Qto_`, ative "Calcular quantidades pela geometria" na barra lateral. Valores convertidos para m, m², m³.
""")