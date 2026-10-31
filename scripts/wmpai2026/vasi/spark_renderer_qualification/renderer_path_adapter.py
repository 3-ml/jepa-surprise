"""Map only the frozen worker's one Desktop-specific scratch literal."""
from pathlib import Path
OLD_SCRATCH=__import__("os").environ['JEPA_FINAL256_SCRATCH']

def path_adapter(scratch, calls):
    def mapped(*parts, **kwargs):
        if len(parts)==1 and isinstance(parts[0],str) and parts[0]==OLD_SCRATCH:
            calls.append({'requested':OLD_SCRATCH,'mapped':str(scratch)})
            return Path(scratch)
        return Path(*parts, **kwargs)
    return mapped
