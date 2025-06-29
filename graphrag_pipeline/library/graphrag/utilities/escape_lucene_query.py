def escape_lucene_query(query: str) -> str:
    """
    Escape special characters in a Lucene query string.
    This function ensures that special characters used in Lucene queries
    are properly escaped to avoid syntax errors or unintended behavior
    (i.e., avoid `org.apache.lucene.queryparser.classic.TokenMgrError`)
    """
    # Lucene special chars that must be escaped
    specials = r'+ - && || ! ( ) { } [ ] ^ " ~ * ? : \ /'.split()
    for s in specials:
        query = query.replace(s, '\\' + s)
    return query