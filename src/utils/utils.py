def parse_labels(s):
    s = s.strip().strip('"').strip("'")  # retire guillemets autour
    return [label.strip() for label in s.split(",") if label.strip()]