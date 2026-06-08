import os
import sys

def may_make_dir(path):
    """
    Args:
        path: a dir, or result of `os.path.dirname(os.path.abspath(file_path))`
    Note:
        `os.path.exists('')` returns `False`, while `os.path.exists('.')` returns `True`!
    """
    # This clause has mistakes:
    # if path is None or '':

    if path in [None, '']:
        return
    if not os.path.exists(path):
        os.makedirs(path)

class ReDirectSTD(object):
    """Modified from Tong Xiao's `Logger` in open-reid.
    This class overwrites sys.stdout or sys.stderr, so that console logs can
    also be written to file.
    Args:
        fpath: file path
        console: one of ['stdout', 'stderr']
        immediately_visible: If `False`, the file is opened only once and closed
            after exiting. In this case, the message written to file may not be
            immediately visible (Because the file handle is occupied by the
            program?). If `True`, each writing operation of the console will
            open, write to, and close the file. If your program has tons of writing
            operations, the cost of opening and closing file may be obvious. (?)
    Usage example:
        `ReDirectSTD('stdout.txt', 'stdout', False)`
        `ReDirectSTD('stderr.txt', 'stderr', False)`
    NOTE: File will be deleted if already existing. Log dir and file is created
        lazily -- if no message is written, the dir and file will not be created.
    """

    def __init__(self, fpath=None, console='stdout', immediately_visible=False):
        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == 'stdout' else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visible = immediately_visible
        # if exists, remove
        if self.f is not None and os.path.exists(self.f):
            os.remove(self.f)
        # Overwrite
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            tmp_dir = os.path.dirname(os.path.abspath(self.file))
            may_make_dir(tmp_dir)
            if self.immediately_visible:
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')
                self.f.write(msg)

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()
