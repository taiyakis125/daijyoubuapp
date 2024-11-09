from huggingface_hub import HfApi, HfFolder, Repository

# アクセストークンを保存してログイン（事前にアクセストークンを生成しておく）
HfFolder.save_token("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")

# リポジトリ指定
repo_url = "https://huggingface.co/hika234/daijyoubuapp"

# リポジトリをローカルにクローン
repo = Repository(
    local_dir="/Users/luccired/Downloads/saved_model_2",  # モデルファイルがあるローカルディレクトリのパス
    clone_from=repo_url,
)

# モデルファイルをリポジトリにアップロード
repo.push_to_hub()
