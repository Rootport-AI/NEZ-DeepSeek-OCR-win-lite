using System;
using System.Diagnostics;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Web.WebView2.Wpf;

namespace NEZ.Shell;

public partial class MainWindow : Window
{
    private Process? _serverProc;
    private readonly AppSettings _settings;
    private readonly Paths _paths;
    private readonly Logger _log;

    public MainWindow()
    {
        InitializeComponent();

        _paths = Paths.InitFromExecutable();
        _log = new Logger(_paths.LogDir);
        _settings = AppSettings.LoadOrDefault(_paths.SettingsPath,
            defaultPort: 8000,
            defaultModelDir: System.IO.Path.Combine(_paths.Root, @"NEZ\DeepSeek-OCR"));
        // ★ WebView2 ユーザーデータを <Root>\NEZ\wv2 に固定（ポータブル運用）
        var wv2Dir = System.IO.Path.Combine(_paths.Root, "NEZ", "wv2");
        System.IO.Directory.CreateDirectory(wv2Dir);
        // WebView2 コントロール名が "webView" の想定（あなたの XAML 名に合わせてください）
        // EnsureCoreWebView2Async の前に CreationProperties を設定しておくことが重要
        Web.CreationProperties = new CoreWebView2CreationProperties
        {
           UserDataFolder = wv2Dir
        };

        this.Loaded += async (_, __) =>
        {
           // 念のため明示的に Core 環境を起こしてからブート（UserDataFolder 反映を確実に）
           try { await Web.EnsureCoreWebView2Async(); } catch { /* 初期化失敗は BootAsync 側の表示で分かる */ }
           await BootAsync();
        };
        this.Closing += (_, __) => Cleanup();
    }

    private async Task BootAsync()
    {
        Overlay.Visibility = Visibility.Visible;
        OverlayDetail.Text = "Python サーバーを起動しています...";

        try
        {
            // 1) 子プロセス起動
            _serverProc = PythonHost.StartServer(_paths, _settings, _log);
            _log.Info("Server process started. PID=" + _serverProc.Id);

            // 2) /healthz 待機
            var ok = await HealthProbe.WaitUntilHealthyAsync(
                $"http://127.0.0.1:{_settings.ServerPort}/healthz",
                TimeSpan.FromSeconds(20),
                _log,
                progress: s => OverlayDetail.Text = s);

            if (!ok)
            {
                _log.Error("/healthz timeout");
                OverlayDetail.Text = "起動に失敗しました（/healthz タイムアウト）。";
                RetryBtn.Visibility = Visibility.Visible;
                ExitBtn.Visibility = Visibility.Visible;
                return;
            }

            // 3) WebView2 へ遷移
            var uri = new Uri($"http://127.0.0.1:{_settings.ServerPort}/");
            Web.Source = uri;
            Overlay.Visibility = Visibility.Collapsed;
        }
        catch (Exception ex)
        {
            _log.Exception(ex, "Boot failed");
            OverlayDetail.Text = "起動に失敗しました：" + ex.Message;
            RetryBtn.Visibility = Visibility.Visible;
            ExitBtn.Visibility = Visibility.Visible;
        }
    }

    private async void RetryBtn_Click(object sender, RoutedEventArgs e)
    {
        RetryBtn.Visibility = Visibility.Collapsed;
        ExitBtn.Visibility = Visibility.Collapsed;
        OverlayDetail.Text = "再試行中...";
        Cleanup();
        await BootAsync();
    }

    private void ExitBtn_Click(object sender, RoutedEventArgs e) => this.Close();

    private void Cleanup()
    {
        try
        {
            if (_serverProc != null && !_serverProc.HasExited)
            {
                _log.Info("Killing server process...");
                _serverProc.Kill(true);
                _serverProc.WaitForExit(5000);
            }
        }
        catch (Exception ex)
        {
            _log.Exception(ex, "Kill failed");
        }
        finally
        {
            _serverProc?.Dispose();
            _serverProc = null;
        }
    }
}
