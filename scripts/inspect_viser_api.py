import inspect
import sys

def main():
    try:
        import viser
        print("viser file:", getattr(viser, "__file__", None))
        print("viser version:", getattr(viser, "__version__", "unknown"))
        try:
            import viser._scene as s
        except Exception as e:
            print("failed to import viser._scene:", repr(e))
            s = None
        if s is not None:
            print("has SceneApi:", hasattr(s, "SceneApi"))
            if hasattr(s, "SceneApi"):
                print("SceneApi add_* methods:")
                for m in dir(s.SceneApi):
                    if m.startswith("add_"):
                        fn = getattr(s.SceneApi, m)
                        try:
                            sig = inspect.signature(fn)
                        except Exception:
                            sig = None
                        print(" -", m, sig)
    except Exception as e:
        print("import viser failed:", repr(e))

if __name__ == "__main__":
    main()


