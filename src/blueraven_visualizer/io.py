"""Blue Raven CSV loading."""

import os
import pandas as pd


class BlueRavenFormatError(Exception):
    """Raised when a CSV doesn't look like a Blue Raven HR/LR export."""


def load_blueraven(path):
    """Read a Blue Raven CSV. Returns (DataFrame, col(name)) where col() maps an
    ORIGINAL header to its column by first occurrence, so the LR file's repeated
    headers don't collide."""
    try:
        raw = pd.read_csv(path, nrows=0).columns.tolist()
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise BlueRavenFormatError(
            f'Could not parse "{os.path.basename(path)}" as a Blue Raven CSV export '
            f"(got: {exc}). Make sure this is the raw HR or LR .csv from the Featherweight "
            f"Blue Raven app, not a re-saved/edited copy."
        ) from exc
    df.columns = range(df.shape[1])          # index by position; names via raw

    def col(name, required=True):
        try:
            i = raw.index(name)
        except ValueError:
            if required:
                raise BlueRavenFormatError(
                    f'Column "{name}" not found in {os.path.basename(path)}. '
                    f"This usually means the wrong file was passed (e.g. an LR file "
                    f"where an HR file was expected, or vice versa), or the export is "
                    f"from a firmware version with a different column layout."
                ) from None
            return None
        return pd.to_numeric(df[i], errors="coerce").to_numpy()

    return df, col
