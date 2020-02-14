package net.finmath.util;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class FileUtils {

	/**
	 * Fully reads the given InputStream and returns it as a string.
	 *
	 * @param inputStream The input stream to read.
	 * @return The contents of the input stream as string.
	 */
	public static String readToString(final InputStream inputStream)
	{
		try(BufferedReader br = new BufferedReader(new InputStreamReader(inputStream)))
		{
			final StringBuffer sb = new StringBuffer();
			String line = null;
			while (true)
			{
				line = br.readLine();
				if (line == null)
				{
					break;
				}
				sb.append(line).append("\n");
			}
			return sb.toString();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
			System.exit(1);
			return null;
		}
	}

	/**
	 * Fully reads the given InputStream and returns it as a byte array.
	 *
	 * @param inputStream The input stream to read
	 * @return The byte array containing the data from the input stream
	 * @throws IOException If an I/O error occurs
	 */
	public static byte[] readToByteArray(final InputStream inputStream) throws IOException
	{
		final ByteArrayOutputStream baos = new ByteArrayOutputStream();
		final byte buffer[] = new byte[8192];
		while (true)
		{
			final int read = inputStream.read(buffer);
			if (read == -1)
			{
				break;
			}
			baos.write(buffer, 0, read);
		}
		return baos.toByteArray();
	}
}
