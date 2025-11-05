using System;
using System.IO;
using System.Reflection;

namespace NEZ.Shell;

public sealed class Paths
{
    public string ExeDir { get; }
    public string Root { get; }              // ...\NEZ-DeepSeekOCR\NEZ.Shell\bin\...\ => 基準に使う
    public string ServerDir { get; }         // ...\NEZ-DeepSeekOCR\server\
    public string ServerAppPy { get; }       // ...\server\app.py
    public string VenvPython { get; }        // ...\runtime\venv\Scripts\python.exe
    public string SettingsPath { get; }      // <Root>\NEZ\settings.json
    public string LogDir { get; }            // <Root>\NEZ\logs

    private Paths(string exeDir, string root)
    {
        ExeDir = exeDir;
        Root = root;
        ServerDir = Path.GetFullPath(Path.Combine(root, "server"));
        ServerAppPy = Path.Combine(ServerDir, "app.py");
        VenvPython = Path.GetFullPath(Path.Combine(root, "runtime", "venv", "Scripts", "python.exe"));

        var nez = Path.Combine(root, "NEZ");
        Directory.CreateDirectory(nez);
        Directory.CreateDirectory(Path.Combine(nez, "logs"));
        SettingsPath = Path.Combine(nez, "settings.json");
        LogDir = Path.Combine(nez, "logs");
    }

    public static Paths InitFromExecutable()
    {
        var exeDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!;
        // exeDir = ...\NEZ.Shell\NEZ.Shell\bin\Release\net8.0-windows...\ など
        // ルートは 「NEZ-DeepSeekOCR」 フォルダ（この一つ下に NEZ\ と server\ が並ぶ想定）
                var root = Path.GetFullPath(Path.Combine(exeDir, @"..\..\..\..\.."));
        return new Paths(exeDir, root);
    }
}
