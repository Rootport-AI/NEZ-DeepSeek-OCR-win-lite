using System;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace NEZ.Shell;

public static class HealthProbe
{
    public static async Task<bool> WaitUntilHealthyAsync(string url, TimeSpan timeout, Logger log, Action<string>? progress = null)
    {
        using var cts = new CancellationTokenSource(timeout);
        using var http = new HttpClient();

        var nextReport = DateTime.MinValue;
        while (!cts.IsCancellationRequested)
        {
            try
            {
                var res = await http.GetAsync(url, cts.Token);
                if (res.IsSuccessStatusCode)
                {
                    // 形式: { "status":"ok", "model":"..." }
                    var json = await res.Content.ReadAsStringAsync(cts.Token);
                    try
                    {
                        var doc = JsonDocument.Parse(json);
                        var status = doc.RootElement.GetProperty("status").GetString();
                        if (status == "ok") return true;
                    }
                    catch { /* JSON不整合でもOK扱いにして良い */ return true; }
                }
            }
            catch { /* ignore and retry */ }

            if (DateTime.Now >= nextReport)
            {
                progress?.Invoke("サーバーの起動を待機中…");
                nextReport = DateTime.Now.AddMilliseconds(500);
            }
            await Task.Delay(250, cts.Token);
        }

        return false;
    }
}
