"""
run_pipeline.py
================
Runs the whole pipeline (Steps 1 through 7) in order, in one command:

    python run_pipeline.py                       # everything
    python run_pipeline.py --steps 1 2 3          # just steps 1-3
    python run_pipeline.py --caseid 1703142       # also run Step 6's
                                                   # single-patient SHAP
                                                   # explanation for this CASEID

Each step is still a completely normal, independently runnable script
(python 01_data_loader.py, etc.) -- this file just chains them together and
saves you typing. Module names starting with a digit aren't valid Python
import names, hence importlib below instead of a plain `import`.
"""

import argparse
import importlib
import time

STEP_MODULES = {
    1: "01_data_loader",
    2: "02_preprocessing",
    3: "03_scoring",
    4: "04_optuna_weight_learning",
    5: "05_train_model",
    6: "06_shap_explain",
    7: "07_phenograph_visualization",
}


def run_step(step_number, **kwargs):
    module_name = STEP_MODULES[step_number]
    print(f"\n{'=' * 70}\nSTEP {step_number}: {module_name}\n{'=' * 70}")
    t0 = time.time()
    module = importlib.import_module(module_name)
    if step_number == 6:
        # Step 6's main() reads argparse's sys.argv, which isn't set when
        # called like this -- call its pieces directly instead so --caseid
        # from THIS script's arguments flows through correctly.
        df, X_scaled, feature_to_variable, feature_to_domain, model_bmi, model_wgt, split = module.load_everything()
        importance_df = module.run_global_explanation(df, X_scaled, feature_to_variable, feature_to_domain,
                                                        model_bmi, model_wgt, split)
        module.build_weight_vs_shap_table(importance_df)
        caseid = kwargs.get("caseid")
        if caseid:
            module.explain_single_patient(caseid)
        else:
            print("[run_pipeline] No --caseid given -- skipping the single-patient SHAP explanation.")
    else:
        module.main()
    print(f"[run_pipeline] Step {step_number} finished in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run the obesity digital-twin phenotype pipeline.")
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEP_MODULES.keys()),
                         help="Which step numbers to run, e.g. --steps 1 2 3. Default: all of them.")
    parser.add_argument("--caseid", type=str, default=None,
                         help="CASEID to generate a single-patient SHAP explanation for, during Step 6.")
    args = parser.parse_args()

    overall_start = time.time()
    for step in sorted(args.steps):
        run_step(step, caseid=args.caseid)
    print(f"\n{'=' * 70}\nPipeline finished in {(time.time() - overall_start) / 60:.1f} minutes.\n{'=' * 70}")


if __name__ == "__main__":
    main()
