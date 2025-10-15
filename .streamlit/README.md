# Streamlit Secrets Setup

This directory contains configuration files for deploying your app to Streamlit Cloud while maintaining local development compatibility.

## For Local Development

1. Copy `secrets.toml.template` to `secrets.toml`:
   ```bash
   cp .streamlit/secrets.toml.template .streamlit/secrets.toml
   ```

2. Edit `secrets.toml` and fill in your actual credential values.

3. The app will automatically use secrets from `secrets.toml` if available, otherwise it will fall back to environment variables.

## For Streamlit Cloud Deployment

1. In your Streamlit Cloud dashboard, go to your app settings.

2. Navigate to the "Secrets" tab.

3. Copy the contents of your `secrets.toml` file into the secrets editor.

4. Deploy your app - it will automatically use the secrets you've configured.

## Security Notes

- `secrets.toml` is already included in `.gitignore` to prevent accidental commits
- Never commit actual credentials to your repository
- Use the template file as a reference for the required structure
- For SERVICE_ACCOUNT_JSON, make sure to properly escape the JSON string

## Environment Variable Fallback

The app supports both secrets and environment variables:

### AWS Credentials
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY` 
- `AWS_REGION`

### Google Cloud Credentials
- `PROJECT_ID`
- `VERTEX_REGION`
- `SERVICE_ACCOUNT_PATH` (path to JSON file)
- `SERVICE_ACCOUNT_JSON` (JSON content as string)

The app will check Streamlit secrets first, then fall back to environment variables if secrets are not available.