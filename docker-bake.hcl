group "default" {
  targets = ["klasyfikator-build"]
}

target "klasyfikator-build" {
  context = "."
  dockerfile = "Dockerfile"
  target = "klasyfikator-build"
  tags = ["ghcr.io/kuball1000/inzynierka_klasyfikator/klasyfikator:latest"]
  platforms = ["linux/amd64"]
  description = "Word embedding transformer for predicting the possible endpoint from Konie API for a given insert query"
  labels = {
    "org.opencontainers.image.source" = "https://github.com/kuball1000/inzynierka_klasyfikator"
  }
}