using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace NEZ.Shell;

public sealed class Logger
{
    private readonly string _logFile;

    public Logger(string logDir)
    {
        var name = DateTime.Now.ToString("yyyyMMdd") + ".log";
        _logFile = Path.Combine(logDir, name);
        Info("===== NEZ Shell start =====");
    }

    public void Info(string msg) => Write("INFO", msg);
    public void Error(string msg) => Write("ERROR", msg);
    public void Exception(Exception ex, string msg) => Write("EXC", $"{msg}: {ex}");

    private void Write(string level, string msg)
    {
        var line = $"{DateTime.Now:HH:mm:ss} [{level}] {msg}";
        try { File.AppendAllText(_logFile, line + Environment.NewLine, Encoding.UTF8); } catch { }
        Debug.WriteLine(line);
    }

    public void AttachProcess(Process p, string tag = "SRV")
    {
        try
        {
            p.OutputDataReceived += (_, e) => { if (e.Data != null) Write(tag, e.Data); };
            p.ErrorDataReceived += (_, e) => { if (e.Data != null) Write(tag + "-ERR", e.Data); };
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
        }
        catch (Exception ex)
        {
            Exception(ex, "AttachProcess failed");
        }
    }
}
