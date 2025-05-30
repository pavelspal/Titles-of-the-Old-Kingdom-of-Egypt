import subprocess
import os

"""
 BIO: 
 This script runs summary files in one execution.
 - Scripts are listed in 'script_list'
"""


if __name__ == '__main__':
    script_list = [
        "summary_persons.py",
        "summary_shap_calculator.py",
        "summary_title.py",
        "summary_model.py",
        "summary_plot_shap_summary.py",
        "summary_plot_shap_mean.py"
    ]

    for script in script_list:
        print(f'=============================================')
        print(f" RUNNING SCRIPT {script}")
        print(f'=============================================')

        result = subprocess.run(["python", script],
                                capture_output=True,
                                text=True)
        print(result.stdout, result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"{script} failed (exit code {result.returncode})")

        print(f'=============================================')
        print(f" FINISHED SCRIPT {script}")
        print(f'=============================================\n\n\n')
    print("All scripts finished successfully.")
