using System;
using System.IO;
using System.Text.Json;

namespace NEZ.Shell;

public sealed class AppSettings
{
    public int ServerPort { get; set; } = 8000;
    public string ModelDir { get; set; } = "";
    public string? DevPythonPath { get; set; } // 開発中にシステムPythonを使う場合に指定

    public static AppSettings LoadOrDefault(string settingsPath, int defaultPort, string defaultModelDir)
    {
        try
        {
            if (File.Exists(settingsPath))
            {
                var json = File.ReadAllText(settingsPath);
                var s = JsonSerializer.Deserialize<AppSettings>(json) ?? new AppSettings();
                if (s.ServerPort <= 0) s.ServerPort = defaultPort;
                if (string.IsNullOrWhiteSpace(s.ModelDir)) s.ModelDir = defaultModelDir;
                return s;
            }
        }
        catch { /* ignore and fall back */ }

        // 初回は既定値でファイル生成
        var def = new AppSettings { ServerPort = defaultPort, ModelDir = defaultModelDir };
        try
        {
            var json = JsonSerializer.Serialize(def, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(settingsPath, json);
        }
        catch { /* ignore */ }
        return def;
    }
}
