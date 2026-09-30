from phishguard.cli import COMMANDS, main


def test_help_lists_every_command(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for cmd in COMMANDS:
        assert cmd in out


def test_unknown_command_returns_2(capsys):
    assert main(["nope"]) == 2


def test_subcommand_modules_import():
    import importlib

    for cmd, (module, _) in COMMANDS.items():
        if module:
            assert hasattr(importlib.import_module(module), "main"), cmd
