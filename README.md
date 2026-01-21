# ContentUnderstanding-RAG

Azure AI Content Understanding を使った動画シーン分割 + Azure AI Search による RAG 検索パイプライン

## 概要 / Overview

このプロジェクトは、動画を**シーン（セグメント）単位**で分析し、各シーンのメタデータ（タグ、説明文、文字起こし）を生成して Azure AI Search に投入することで、LLM からの検索（RAG）を可能にするパイプラインです。

**主な機能:**
- Azure AI Content Understanding で動画を自動的にシーン分割（`enableSegment=true`）
- 各シーンごとにタグ、説明文、文字起こしを生成
- 1シーン = 1ドキュメントとして Azure AI Search にインデックス
- REST API（`/query`）でベクター検索 + フィルタによる関連シーン検索
- 検索結果に動画URL + 時間範囲（start/end）を含めて返却

## アーキテクチャ / Architecture

```
Video (Blob SAS URL)
    ↓
[Azure AI Content Understanding]
    ↓ (enableSegment=true)
Scene Segments (tags, description, transcript)
    ↓
[Normalization] → SceneDocument[]
    ↓
[Embedding Generation] (Azure OpenAI)
    ↓
[Azure AI Search] (Index: 1 scene = 1 doc)
    ↓
[Query API] (/query endpoint)
    ↓
RAG Results (video URL + time range)
```

## 必要な Azure リソース / Prerequisites

1. **Azure AI Content Understanding**
   - Endpoint URL
   - Subscription Key
   - API Version: 2025-11-01 以降

2. **Azure AI Search**
   - Endpoint URL
   - Admin API Key
   - インデックス名（デフォルト: `video-scenes`）

3. **Azure OpenAI** (埋め込み生成用、Optional)
   - Endpoint URL
   - API Key
   - Deployment: `text-embedding-3-small` または類似モデル

## セットアップ / Setup

### 1. リポジトリをクローン

```bash
git clone https://github.com/nao-kat/ContentUnderstanding-RAG.git
cd ContentUnderstanding-RAG
```

### 2. Python 環境のセットアップ

```bash
# Python 3.10+ 推奨
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 3. 環境変数の設定

`.env.example` をコピーして `.env` を作成し、Azure リソースの情報を設定します。

```bash
cp .env.example .env
```

`.env` ファイルを編集:

```bash
# Azure AI Content Understanding
CU_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
CU_KEY=<your-key>
CU_API_VERSION=2025-11-01

# Azure AI Search
SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
SEARCH_INDEX_NAME=video-scenes
SEARCH_API_KEY=<your-search-admin-key>

# Azure OpenAI (for embeddings)
EMBEDDING_ENDPOINT=https://<your-openai-resource>.openai.azure.com
EMBEDDING_KEY=<your-openai-key>
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
```

## 使い方 / Usage

### 動画の取り込み（Ingest）

動画を分析してシーンをインデックスに登録します。

```bash
python -m src.ingest --video_url "https://your-storage.blob.core.windows.net/videos/sample.mp4?<SAS-token>"
```

**オプション:**
- `--video_id`: 動画ID（省略時はURLから自動抽出）
- `--no-embeddings`: 埋め込み生成をスキップ

**処理フロー:**
1. Azure AI Content Understanding で動画を分析
2. セグメント（シーン）を抽出
3. 各シーンを SceneDocument に正規化
4. 埋め込みベクトルを生成（Optional）
5. Azure AI Search にインデックス

**実行例:**

```bash
$ python -m src.ingest --video_url "https://example.blob.core.windows.net/videos/demo.mp4?sv=..."

Starting video ingestion: https://example.blob.core.windows.net/videos/demo.mp4?sv=...
Video ID: demo

[1/5] Analyzing video with Azure AI Content Understanding...
✓ Video analysis complete

[2/5] Parsing segments from analysis results...
✓ Found 5 segment(s)

[3/5] Normalizing segments to scene documents...
✓ Created 5 scene document(s)

[4/5] Generating embeddings...
✓ Generated embeddings for 5 document(s)

[5/5] Indexing documents in Azure AI Search...
  Created new search index
✓ Indexed 5 document(s)

============================================================
INGESTION COMPLETE
============================================================
Video ID: demo
Segments: 5
Documents indexed: 5
Search index: video-scenes

You can now query scenes using the /query API endpoint
```

### クエリ API の起動

FastAPI サーバーを起動します。

```bash
python -m src.api
```

または uvicorn で直接起動:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

サーバーが起動したら、ブラウザで `http://localhost:8000/docs` を開くと Swagger UI が表示されます。

### シーンの検索（Query）

#### REST API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "person walking in a park",
    "top_k": 5
  }'
```

**レスポンス例:**

```json
{
  "results": [
    {
      "video_url": "https://example.blob.core.windows.net/videos/demo.mp4",
      "video_id": "demo",
      "start_time_ms": 123000,
      "end_time_ms": 156000,
      "tags": ["outdoor", "person", "park"],
      "scene_description": "A person is walking through a sunny park with trees.",
      "transcript_text": "This is a beautiful day for a walk.",
      "score": 0.87
    }
  ],
  "query": "person walking in a park",
  "count": 1
}
```

#### フィルタオプション

```bash
# 特定の動画に絞る
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "action scene",
    "top_k": 3,
    "video_id": "demo"
  }'

# タグでフィルタ
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "outdoor activities",
    "top_k": 5,
    "tag_filter": ["outdoor", "sport"]
  }'
```

## テスト / Testing

```bash
# すべてのテストを実行
pytest

# 特定のテストファイルのみ
pytest tests/test_normalize.py
pytest tests/test_query_contract.py

# カバレッジ付き
pytest --cov=src tests/
```

## プロジェクト構成 / Project Structure

```
ContentUnderstanding-RAG/
├── src/
│   ├── __init__.py
│   ├── config.py              # 環境変数管理
│   ├── cu_client.py           # Content Understanding API クライアント
│   ├── cu_parse.py            # CU レスポンスのパース
│   ├── normalize.py           # SceneDocument への正規化
│   ├── embedding.py           # 埋め込みベクトル生成
│   ├── search_index.py        # Azure AI Search インデックス管理
│   ├── api.py                 # FastAPI アプリケーション（/query）
│   └── ingest.py              # CLI: 動画取り込みツール
├── tests/
│   ├── __init__.py
│   ├── test_normalize.py      # 正規化ロジックのテスト
│   └── test_query_contract.py # API レスポンス契約のテスト
├── infra/
│   └── index_schema.json      # Search インデックススキーマ定義
├── .env.example               # 環境変数テンプレート
├── .gitignore
├── requirements.txt           # Python 依存パッケージ
├── LICENSE
└── README.md                  # このファイル
```

## SceneDocument スキーマ

各シーンは以下のフィールドを持つドキュメントとして保存されます:

```json
{
  "id": "videoId-seg0001",
  "video_id": "videoId",
  "video_url": "https://...",
  "start_time_ms": 123000,
  "end_time_ms": 156000,
  "tags": ["outdoor", "person"],
  "scene_description": "A person walking in a park",
  "transcript_text": "This is a sunny day.",
  "content_text": "A person walking in a park\nThis is a sunny day.",
  "content_vector": [0.1, 0.2, ..., 0.9]  // 1536次元
}
```

## トラブルシューティング / Troubleshooting

### CU API がタイムアウトする

- 動画のサイズが大きい場合、分析に時間がかかります
- `cu_client.py` の `timeout` を増やしてください（デフォルト: 600秒）

### 埋め込み生成に失敗する

- `EMBEDDING_ENDPOINT` と `EMBEDDING_KEY` が正しく設定されているか確認
- `--no-embeddings` オプションで埋め込みをスキップして動作確認

### Search インデックスが作成できない

- `SEARCH_API_KEY` が Admin Key であることを確認（Query Key では不可）
- Azure AI Search のリソースが正常に動作しているか確認

## ライセンス / License

MIT License - 詳細は [LICENSE](LICENSE) を参照

## 貢献 / Contributing

Issue や Pull Request を歓迎します！

## 参考資料 / References

- [Azure AI Content Understanding Documentation](https://learn.microsoft.com/azure/ai-services/content-understanding/)
- [Azure AI Search Documentation](https://learn.microsoft.com/azure/search/)
- [Azure OpenAI Service Documentation](https://learn.microsoft.com/azure/ai-services/openai/)