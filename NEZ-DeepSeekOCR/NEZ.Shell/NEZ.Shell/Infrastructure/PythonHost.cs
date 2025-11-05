using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace NEZ.Shell
{
    public static class PythonHost
    {
        public static Process StartServer(Paths paths, AppSettings s, Logger log)
        {
            // 1) 起動ターゲットを決定
            //    まずは PyInstaller の onedir 生成物（コンソールあり）を優先して直起動する。
            //    無ければ従来どおり python.exe で app.py を起動（開発用フォールバック）。
            var serverDir = Path.GetDirectoryName(paths.ServerAppPy)!;                 // ...\server
            var distDir = Path.Combine(serverDir, "dist", "NEZ.Server");
            var serverExe = Path.Combine(distDir, "NEZ.Server.exe");

            bool useExe = File.Exists(serverExe);
            string launcher = useExe
                ? serverExe
                : (!string.IsNullOrWhiteSpace(s.DevPythonPath) && File.Exists(s.DevPythonPath)
                    ? s.DevPythonPath
                    : paths.VenvPython);

            if (!File.Exists(launcher))
                throw new FileNotFoundException(useExe ? "NEZ.Server.exe が見つかりません" : "python.exe が見つかりません", launcher);

            if (!useExe && !File.Exists(paths.ServerAppPy))
                throw new FileNotFoundException("server\\app.py が見つかりません", paths.ServerAppPy);

            // 2) プロセス起動情報
            var psi = new ProcessStartInfo
            {
                FileName = launcher,
                Arguments = useExe ? "" : $"\"{paths.ServerAppPy}\"",
                WorkingDirectory = useExe ? distDir : Path.GetDirectoryName(paths.ServerAppPy)!,
                UseShellExecute = false,

                // コンソール“あり”ビルドを前提：黒い画面に直接ログを出したいのでリダイレクトしない
                RedirectStandardOutput = false,
                RedirectStandardError = false,
                CreateNoWindow = false,

                // （必要になったら復活）
                // StandardOutputEncoding = Encoding.UTF8,
                // StandardErrorEncoding  = Encoding.UTF8,
            };

            // 3) 子プロセス専用の環境変数
            //    EXE直起動時は venv の PATH 注入は不要。python起動時のみ venv を最優先にする。
            if (!useExe)
            {
                var venvRoot = Path.GetFullPath(Path.Combine(paths.Root, "runtime", "venv"));
                var venvScripts = Path.Combine(venvRoot, "Scripts");
                var oldPath = Environment.GetEnvironmentVariable("PATH") ?? "";
                psi.Environment["PATH"] = $"{venvScripts};{venvRoot};{oldPath}";
                psi.Environment["VIRTUAL_ENV"] = venvRoot;      // 念のため venv を明示
                psi.Environment["PYTHONIOENCODING"] = "utf-8";  // print()/logging を UTF-8 に
                psi.Environment["PYTHONUTF8"] = "1";            // 追加の UTF-8 フラグ（3.7+）
            }

            // モデルや実行方針は EXE/py どちらの場合も共通で渡す
            psi.Environment["DEEPSEEK_OCR_MODEL"] = s.ModelDir;
            psi.Environment["USE_TF"] = "0";
            psi.Environment["USE_TORCH"] = "1";
            // 既定は 127.0.0.1:8000 のまま。将来 app.py 側で PORT を読むようにしたら渡す：
            // psi.Environment["PORT"] = s.ServerPort.ToString();

            // 4) 起動
            var p = Process.Start(psi)!;

            // コンソールに直接出す運用のため、標準出力の取り込みは一旦不要。
            // 必要になれば条件分岐のうえで Attach を呼び戻す。
            // log.AttachProcess(p);

            return p;
        }
    }
}
