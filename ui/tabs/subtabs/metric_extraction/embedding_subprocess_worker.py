#!/usr/bin/env python3
import sys
import os
import json

# Force PyTorch to use 1 thread to avoid RHEL 7 Segmentation Faults
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def main():
    # ── 1. Read parameters from stdin ────────────────────────────────
    try:
        input_data = sys.stdin.read()
        if not input_data:
            return
        params = json.loads(input_data)
    except Exception as exc:
        _emit_error(f"Failed to parse input parameters: {exc}")
        sys.exit(1)

    input_path    = params["input_path"]
    output_path   = params["output_path"]
    libclang_path = params["libclang_path"]
    model_path    = params["model_path"]
    app_root      = params.get("app_root", "")

    # Inject app_root into sys.path so we can find the 'core' package
    if app_root and app_root not in sys.path:
        sys.path.insert(0, app_root)

    # ── 2. Phase A: libclang — extract functions ──────────────────────
    try:
        # UPDATED: Importing from the new 'core' location
        from core.metrics_extractor import MetricsExtractor,initialize_clang_library
        initialize_clang_library(libclang_path)
        extractor = MetricsExtractor(libclang_path)
    except ImportError:
        
        _emit_error(f"Could not find MetricsExtractor in 'core': {exc}")
        sys.exit(1)
    except Exception as exc:
        _emit_error(f"libclang initialisation failed: {exc}")
        sys.exit(1)

    if not os.path.isdir(input_path):
        _emit_error(f"Input folder not found: {input_path}")
        sys.exit(1)

    try:
        # This assumes your core extractor uses the same method name
        functions_list = extractor.extract_functions_from_folder(input_path)
    except Exception as exc:
        _emit_error(f"Function extraction failed: {exc}")
        sys.exit(1)

    if not functions_list:
        _emit_error("No functions found. Ensure folder contains .c / .cpp files.")
        sys.exit(1)

    # Prepare data for PyTorch
    code_strings = []
    function_names = []
    for func_data in functions_list:
        code = func_data.get("fCode", "").strip()
        if code:
            code_strings.append(code)
            function_names.append(func_data.get("Function", "unknown"))

    # Crucial: Cleanup libclang before loading heavy Torch libraries
    del extractor
    
    # ── 3. Phase B: PyTorch — generate embeddings ────────────────────
    try:
        import torch
        import pandas as pd
        from transformers import AutoTokenizer, AutoModel
    except Exception as exc:
        _emit_error(f"Failed to import PyTorch: {exc}")
        sys.exit(1)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # local_files_only=True ensures we use your local Desktop path
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model     = AutoModel.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        _emit_error(f"Model Load Error: {exc}")
        sys.exit(1)

    embeddings = []
    meta_names = []
    total = len(code_strings)

    for i, (code, fname) in enumerate(zip(code_strings, function_names)):
        try:
            inputs = tokenizer(
                code, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                
                # Masked Mean Pooling logic (Ignoring padding for accuracy)
                last_hidden = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                
                sum_embeddings = torch.sum(last_hidden * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                mean_pooled = (sum_embeddings / sum_mask).squeeze()
                
                embeddings.append(mean_pooled.cpu().numpy())
                meta_names.append(fname)
                
        except Exception as inner:
            sys.stderr.write(f"Skipping {fname}: {inner}\n")

        _emit_progress(i + 1, total)

    # ── 4. Write Output ──────────────────────────────────────────────
    if not embeddings:
        _emit_error("No embeddings generated.")
        sys.exit(1)

    try:
        emb_df = pd.DataFrame(embeddings, columns=[f"emb_{j}" for j in range(len(embeddings[0]))])
        meta_df = pd.DataFrame({"function_name": meta_names})
        final_df = pd.concat([meta_df, emb_df], axis=1)
        final_df.to_csv(output_path, index=False)
    except Exception as exc:
        _emit_error(f"CSV Write Error: {exc}")
        sys.exit(1)

    _emit({"type": "done", "rows": len(final_df)})

# ── Helpers ──────────────────────────────────────────────────────────

def _emit(obj: dict):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def _emit_progress(current: int, total: int):
    _emit({"type": "progress", "current": current, "total": total})

def _emit_error(message: str):
    _emit({"type": "error", "message": message})

if __name__ == "__main__":
    main()