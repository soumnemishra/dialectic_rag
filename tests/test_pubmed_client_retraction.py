from bs4 import BeautifulSoup

from src.pubmed_client import PubMedClient


def test_pubmed_comments_corrections_sets_retraction_flag():
    xml = """
    <PubmedArticle>
      <MedlineCitation>
        <PMID>999</PMID>
        <Article>
          <ArticleTitle>Trial title</ArticleTitle>
          <Abstract><AbstractText>We enrolled 100 participants.</AbstractText></Abstract>
          <Journal>
            <Title>Journal</Title>
            <JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue>
          </Journal>
          <PublicationTypeList>
            <PublicationType>Randomized Controlled Trial</PublicationType>
          </PublicationTypeList>
        </Article>
        <MeshHeadingList>
          <MeshHeading><DescriptorName>Humans</DescriptorName></MeshHeading>
        </MeshHeadingList>
        <CommentsCorrectionsList>
          <CommentsCorrections RefType="RetractionOf"><PMID>111</PMID></CommentsCorrections>
        </CommentsCorrectionsList>
      </MedlineCitation>
    </PubmedArticle>
    """
    article_elem = BeautifulSoup(xml, "xml").find("PubmedArticle")
    client = PubMedClient()

    raw = client._parse_single_article(article_elem)
    assert raw.is_retracted is True

    docs = client.validate_articles([raw])
    assert docs[0].is_retracted is True
