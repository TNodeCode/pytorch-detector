def log_epoch(filepath: str, epoch_metrics: dict):
    # Create logfile lines
    lines = [f"  {k}: {epoch_metrics[k]}" for k in epoch_metrics.keys()]
    # Make object a YAML array
    lines[0] = "\n-" + lines[0][1:]
    # Create YAML object text
    yaml_text = "\n".join(lines)
    with open(filepath, "a") as myfile:
        myfile.write(yaml_text)