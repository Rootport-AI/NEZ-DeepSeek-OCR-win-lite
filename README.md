# NEZ-DeepSeek-OCR-win-lite

![NEZ-DSOCR-winlite](https://github.com/user-attachments/assets/6808e3a3-a62e-4801-8331-fcba5d15343b)

画像認識AI「DeepSeek-OCR」を手軽に動かすGUIです。  
主な使用目的として、スマートフォン等で撮影したレシートの画像のテキスト化を想定しています。  
[以前作成したWEBアプリ](https://github.com/Rootport-AI/NEZ-DeepSeek-OCR-GUI)のWindows版。  
非エンジニア向けに簡単に動かせることを目指しました（NEZ = Non-Engineer's Zapper）。  

# DeepSeek-OCRとは？  
中国企業DeepSeek社の開発した、オープンウェイトの画像認識AIです。ローカル環境で（つまり、オフラインのパソコンで）実行できます。ChatGPTやGeminiとは異なり、読み取った画像やテキストのデータを**外部に送信しません**。

DeepSeek-OCRのAI本体のデータおよび詳細な情報は、HuggingFaceの公式リポジトリで公開されています。（※HuggingFaceとは、AI研究者やAI技術者の情報交流サイトです） https://huggingface.co/deepseek-ai/DeepSeek-OCR

# 動作検証環境
- OS: Windows 11  
  Windows 10でも動くかもしれませんが未検証です。Mac、Linuxは非対応です。
- GPU: RTX 4070 Ti SUPER  
  現状ではVRAM 12GB以上のGPUが推奨です。低VRAMマシンやCPU演算は、将来的に対応するかも。
- ストレージ：**最低 18GB**の空き容量。推奨20GB。
  アプリ本体が5GB超、AIモデルが12GB超あります。

# インストール方法
[![HuggingFaceDSOCRwinlite](https://github.com/user-attachments/assets/70a53f3a-d117-4740-9edb-c244503b88dd)](https://huggingface.co/datasets/Rootport/NEZ-DeepSeek-OCR-win-lite/tree/main)

1. 私の[HuggingFaceリポジトリ](https://huggingface.co/datasets/Rootport/NEZ-DeepSeek-OCR-win-lite/tree/main)を開いて、`NEZ-DSOCR-winlite.zip`(アプリ本体)、および`DeepSeek-OCR.zip`(AI本体)をダウンロードします。  
2. インストールしたいディレクトリで`NEZ-DSOCR-winlite.zip`を展開してください。  
3. `DeepSeek-OCR.zip`を展開し、`\NEZ-DeepSeek-OCR-win-lite\NEZ"`に配置してください。  
4. `NEZ-DSOCR-winlite.exe`をダブルクリックするとアプリが起動します。
   （※初回起動時は非常に時間がかかります。起動に5分間、OCR開始までに2～3分間ほどかかります。）

**注意:** アプリの起動時に一緒に立ち上がる「黒い画面」は、**閉じないでください。** デバッグ用のログがここに表示されます。アプリのウィンドウを閉じると、この「黒い画面」も一緒に閉じます。  

### ▼展開後のディレクトリ構成のイメージ▼  
```
(任意のディレクトリ)\NEZ-DSOCR-winlite
├─NEZ-DSOCR-winlite
│      NEZ-DSOCR-winlit.exe ←アプリ本体
│
├─asset
├─NEZ
│ 　　  DeepSeek-OCR ←★ここに解凍したAI本体をフォルダごと置く
│      settings.json.txt
│
├─NEZ.Shell 
└─server
    │  app.py
    │
    ├─build 
    ├─dist 
    └─static
            index.html
            main.js
            style.css
```

# 作者・ライセンス

- Rootport
  https://x.com/rootport
- MIT license

---

# DeepSeek-OCR.zip について**

- 本アーカイブは、2025-11-05 時点で `https://huggingface.co/deepseek-ai/DeepSeek-OCR` をクローンした**スナップショット**です。自動更新はされません。
- **ライセンス：MIT**（上流の `LICENSE` を同梱／表記を保持）。著作権は DeepSeek に帰属します。
- 内容は**原則無改変**です（同梱物の配置のみ）。将来、差分が生じた場合は本READMEに変更点を記載します。
- 使い方：アプリのルート直下にある `NEZ` フォルダへ **`NEZ\DeepSeek-OCR\`** として展開してください。

  - 既定パス：`NEZ\DeepSeek-OCR`
  - 変更したい場合：`NEZ\settings.json` の `ModelDir`、または環境変数 `DEEPSEEK_OCR_MODEL` で指定可。
- 公式配布元で最新版を取得したい場合は、上記 Hugging Face ページから直接ダウンロードしてください。
- 本パッケージは**非公式ミラー**であり、DeepSeek とは無関係です。名称・ロゴ等は各権利者に帰属します。
- DeepSeek社および開発チームによる DeepSeek-OCR の公開と継続的な開発・保守に深く感謝します。





