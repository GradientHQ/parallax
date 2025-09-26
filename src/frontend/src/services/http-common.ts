export interface HttpCommonOptions {
  /**
   * The URL to fetch.
   */
  url: string | URL;

  /**
   * The HTTP method to fetch.
   * @default "GET"
   */
  method?: "GET" | "POST";

  /**
   * The callback function to process the data after fetch.
   * @param data The data of the fetch response.
   * @returns The new reference data that has been processed.
   */
  afterFetch?: (data: any) => any;
}
