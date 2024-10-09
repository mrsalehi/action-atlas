
import numpy as np
from loguru import logger

from action_atlas.utils import (
    read_jsonl,
    write_jsonl,
)

def bootstrap_confidence_interval_accuracy(
    y_true, 
    y_pred, 
    num_bootstrap=1000):
    """
    Calculate 95% confidence interval for the overall accuracy using the bootstrap method.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Model predicted labels.
    num_bootstrap (int): Number of bootstrap samples to use for calculating confidence intervals.

    Returns:
    tuple: (lower_bound, upper_bound) for the 95% confidence interval of the accuracy.
    """
    n = len(y_pred)
    bootstrap_accuracies = np.zeros(num_bootstrap)
    for i in range(num_bootstrap):
        # Resample indices
        indices = np.random.choice(range(n), size=n, replace=True)
        # Resample true and predicted labels
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
 
        # Calculate accuracy for the resampled data
        bootstrap_accuracies[i] = np.mean(y_pred_bootstrap == y_true_bootstrap)

    # Calculate the 95% confidence interval for accuracy
    lower_bound = np.percentile(bootstrap_accuracies, 2.5)
    upper_bound = np.percentile(bootstrap_accuracies, 97.5)

    mean_accuracy = np.mean(bootstrap_accuracies)
    margin_of_error = mean_accuracy - lower_bound

    plus_minus = u"\u00B1"
    logger.info(f"Mean acc. with margin of error: {mean_accuracy * 100:.2f} {plus_minus} {margin_of_error*100:.2f}")
    logger.info(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")

    return mean_accuracy, lower_bound, upper_bound, margin_of_error


def compute_final_accuracy(
    result_jsonl_fpath: str, 
    verbose: bool=False, 
    conf_interval: bool=False
):
    results = read_jsonl(result_jsonl_fpath)
    errors = 0
    if conf_interval:
        y_true = []
        y_pred = []

    for it, r in enumerate(results, 1):
        if "error" in r:
            r["correct"] = 0
            errors += 1
            response = "-1"
            if verbose:
                logger.info(f"{it}. ground truth: {str(r['gt_answer'])}, response: {response}")
        else:
            response = str(r["response"]).strip() 
            response = "".join(filter(str.isdigit, response)) # remove any non digit characters
            if len(response) > 1:
                response = response[0]
            if verbose:
                logger.info(f"{it}. ground truth: {str(r['gt_answer'])}, response: {response}")

        if not conf_interval:
            if response == str(r["gt_answer"]):
                r["correct"] = 1
            else:
                r["correct"] = 0
        else:
            y_true.append(str(r["gt_answer"]))
            y_pred.append(response)
    
    logger.info(f"result_jsonl_fpath: {result_jsonl_fpath}") 
    logger.info(f"Errors: {errors}")

    if not conf_interval:    
        acc = sum([r['correct'] for r in results]) / len(results)
        logger.info(f"Accuracy for {result_jsonl_fpath}: {acc}")

        return acc

    res = bootstrap_confidence_interval_accuracy(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
    )

    return res


def compute_final_accuracy_with_cot_reasoning(
    result_jsonl_fpath: str, 
    verbose: bool=False, 
    conf_interval: bool=False):
    results = read_jsonl(result_jsonl_fpath)

    if conf_interval:
        y_true = []
        y_pred = []

    for it, r in enumerate(results, 1):
        if "error" in r:
            r["correct"] = 0
            errors += 1
            response = "-1"
            if verbose:
                logger.info(f"{it}. ground truth: {str(r['gt_answer'])}, response: {response}") 
        else:
            response = str(r["response"]).split("\n")[-1]
            # remove any non digit characters
            response = "".join(filter(str.isdigit, response))
            if len(response) > 1:
                response = response[0]
            elif len(response) == 0:
                # Check if the model has outputted only one of the action names
                choices = r["choices"]
                if isinstance(choices, str):
                    choices = choices.split("\n")
                    choices = [opt[len("1. "):].lower() for opt in choices]  # strip off the choice index
                response = str(r["response"]).lower()
                found_option_idx = None
                for idx, opt in enumerate(choices, 1):
                    if opt in response:
                        if found_option_idx is None:
                            found_option_idx = idx
                        else:
                            found_option_idx = None
                            break
                response = "" if found_option_idx is None else str(found_option_idx)
                # last resort: check if only a single digit exists in the response which we assume is the answer
                if len(response) == 0:
                    response = str(r["response"])
                    # remove any non digit characters
                    response = "".join(filter(str.isdigit, response))
                    # check if the response is a single digit
                    if len(response) == 1:
                        response = response[0]
                    else:
                        response = ""
 
        if verbose:
            logger.info(f"{it}. ground truth: {str(r['gt_answer'])}, response: {response}")
        
        if not conf_interval:
            if response == str(r["gt_answer"]):
                r["correct"] = 1
            else:
                r["correct"] = 0
        else:
            y_true.append(str(r["gt_answer"]))
            y_pred.append(response)
    
    if not conf_interval:    
        acc = sum([r['correct'] for r in results]) / len(results)
        logger.info(f"Accuracy: {acc}")
        return acc

    res = bootstrap_confidence_interval_accuracy(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
    )

    return res